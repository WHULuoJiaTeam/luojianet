# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Define the grad rules of math related operations."""

from functools import reduce
import numpy as np
import mindspore as ms
from mindspore import nn
from .. import functional as F
from .. import operations as P
from ..operations import _grad_ops as G
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from ..functional import broadcast_gradient_args, reduced_shape, tuple_div
from .grad_base import bprop_getters
from ..primitive import constexpr
from ..composite.multitype_ops import _constexpr_utils as const_utils
from ..operations._inner_ops import DynamicStitch, DynamicBroadcastGradientArgs, DynamicBroadcastTo
from ...common import Tensor
from .._utils.utils import is_shape_unknown
from ...common import dtype as mstype

shape_op = P.Shape()
dyn_shape_op = P.TensorShape()
reduce_prod = P.ReduceProd()
reduce_sum = P.ReduceSum()
reshape = P.Reshape()
tile = P.Tile()
is_sub_class = P.IsSubClass()
to_array = P.TupleToArray()
real_div = P.RealDiv()


def dyn_binop_grad_common(x, y, dx, dy):
    """
    Common grad definition for binary operations when the input is dynamic shape.

    The function is usually used in backprop op to reduce additional dimensions created by broadcasting.
    """
    shape_of_x = dyn_shape_op(x)
    shape_of_y = dyn_shape_op(y)
    rx, ry = DynamicBroadcastGradientArgs()(shape_of_x, shape_of_y)
    dx = reduce_sum(dx, rx)
    dy = reduce_sum(dy, ry)
    reduce_dx = reshape(dx, shape_of_x)
    reduce_dy = reshape(dy, shape_of_y)
    return reduce_dx, reduce_dy


def dyn_binop_grad_common_with_shift(x, y, dx, dy, shift):
    """
    Common grad definition for binary operations with shift when the input is dynamic shape.

    The function is usually used in backprop op to reduce additional dimensions created by broadcasting.
    """
    shape_of_x = dyn_shape_op(x)
    shape_of_y = dyn_shape_op(y)
    broadcast_shape_of_x = shape_of_x[:-shift]
    broadcast_shape_of_y = shape_of_y[:-shift]
    rx, ry = DynamicBroadcastGradientArgs()(broadcast_shape_of_x, broadcast_shape_of_y)
    dx = reduce_sum(dx, rx)
    dy = reduce_sum(dy, ry)
    reduce_dx = reshape(dx, shape_of_x)
    reduce_dy = reshape(dy, shape_of_y)
    return reduce_dx, reduce_dy


def _reduce_sum_with_cast(dx, axis):
    dx_origin_dtype = dx.dtype
    # Currently, for Ascend and GPU, the reduce_sum's input does not support int16, int32 and int64.
    if dx_origin_dtype in (mstype.int16, mstype.int32, mstype.int64):
        dx = F.cast(dx, mstype.float32)
        dx = reduce_sum(dx, axis)
        return F.cast(dx, dx_origin_dtype)
    return reduce_sum(dx, axis)


def binop_grad_common(x, y, dx, dy):
    """
    Common grad definition for binary operations.

    The function is usually used in backprop op to reduce additional dimensions created by broadcasting.
    """
    shape_of_x = shape_op(x)
    shape_of_y = shape_op(y)
    # if input shape is the same as dout shape, do not need to reduce
    reduce_dx = dx
    reduce_dy = dy
    if not (is_shape_unknown(shape_of_x) or is_shape_unknown(shape_of_y)):
        rx = broadcast_gradient_args(shape_of_x, shape_of_y)
        if rx[0]:
            # if dx is scalar whose shape is (), do not need reduce
            if shape_op(dx):
                dx = _reduce_sum_with_cast(dx, rx[0])
            reduce_dx = reshape(dx, shape_of_x)
        if rx[1]:
            # if dy is scalar whose shape is (), do not need reduce
            if shape_op(dy):
                dy = _reduce_sum_with_cast(dy, rx[1])
            reduce_dy = reshape(dy, shape_of_y)
        return reduce_dx, reduce_dy
    if not shape_of_x or not shape_of_y:
        # x or y is scalar
        if not shape_of_x:
            reduce_dx = _reduce_sum_with_cast(dx, ())
        if not shape_of_y:
            reduce_dy = _reduce_sum_with_cast(dy, ())
        return reduce_dx, reduce_dy

    return dyn_binop_grad_common(x, y, dx, dy)


def binop_grad_common_with_shift(x, y, dx, dy, shift):
    """
    Common grad definition for binary operations with shift.

    The function is usually used in backprop op to reduce additional dimensions created by broadcasting.
    """
    shape_of_x = shape_op(x)
    shape_of_y = shape_op(y)
    broadcast_shape_of_x = shape_of_x[:-shift]
    broadcast_shape_of_y = shape_of_y[:-shift]
    # if input shape is the same as dout shape, do not need to reduce
    reduce_dx = dx
    reduce_dy = dy
    if not (is_shape_unknown(broadcast_shape_of_x) or is_shape_unknown(broadcast_shape_of_y)):
        rx = broadcast_gradient_args(broadcast_shape_of_x, broadcast_shape_of_y)
        if rx[0]:
            # if dx is scalar whose shape is (), do not need reduce
            if shape_op(dx):
                dx = _reduce_sum_with_cast(dx, rx[0])
            reduce_dx = reshape(dx, shape_of_x)
        if rx[1]:
            # if dy is scalar whose shape is (), do not need reduce
            if shape_op(dy):
                dy = _reduce_sum_with_cast(dy, rx[1])
            reduce_dy = reshape(dy, shape_of_y)
        return reduce_dx, reduce_dy
    if not shape_of_x or not shape_of_y:
        # x or y is scalar
        if not shape_of_x:
            reduce_dx = _reduce_sum_with_cast(dx, ())
        if not shape_of_y:
            reduce_dy = _reduce_sum_with_cast(dy, ())
        return reduce_dx, reduce_dy

    return dyn_binop_grad_common_with_shift(x, y, dx, dy, shift)


def _dyn_reduced_shape(input_shape, axis):
    """Dynamic reduce shape"""
    input_shape = P.Cast()(input_shape, ms.int32)
    if isinstance(axis, Tensor):
        input_rank = P.Rank()(input_shape)
        real_axis = (axis + input_rank) % input_rank
        axis_shape = shape_op(real_axis)
    else:
        real_axis = ()
        input_rank = len(input_shape)
        if isinstance(axis, int):
            axis = (axis,)
        elif not axis:
            axis = range(input_rank)
        for i in axis:
            real_axis += ((i + input_rank) % input_rank,)
        axis_shape = (len(real_axis),)
    return DynamicStitch()([to_array(range(input_rank)), to_array(axis)],
                           [input_shape, P.Fill()(ms.int32, axis_shape, 1)])


def _sum_grad(x, axis, dout):
    """Grad definition for `Sum` operation."""
    input_shape = shape_op(x)
    if is_shape_unknown(input_shape):
        input_shape = dyn_shape_op(x)
        output_shape_kept_dims = _dyn_reduced_shape(input_shape, axis)
        grad = reshape(dout, output_shape_kept_dims)
        return DynamicBroadcastTo()(grad, input_shape)

    output_shape_kept_dims = reduced_shape(input_shape, axis)
    tile_scaling = tuple_div(input_shape, output_shape_kept_dims)
    grad = reshape(dout, output_shape_kept_dims)
    return tile(grad, tile_scaling)


def _min_or_max_grad(x, axis, out, dout):
    """Grad definition for `Min` and `Max` operations."""
    input_shape = shape_op(x)
    output_shape_kept_dims = reduced_shape(input_shape, axis)
    y = reshape(out, output_shape_kept_dims)
    grad = reshape(dout, output_shape_kept_dims)
    indicators = F.cast(F.equal(y, x), F.dtype(grad))
    min_num = F.cast(F.scalar_to_array(1e-24), F.dtype(grad))
    num_selected = reshape(reduce_sum(indicators, axis), output_shape_kept_dims) + min_num
    return indicators / num_selected * grad


def _argmin_or_argmax_grad(x, axis, keep_dims, op, out, dout):
    """ArgMinWiwhValue and ArgMaxWithValue grad."""
    expand = P.ExpandDims()
    x_shape = F.shape(x)
    x_dim = len(x_shape)
    x_axis = axis
    if x_axis < 0:
        x_axis = axis + x_dim
    onehot_axis = x_axis
    depth = x_shape[x_axis]
    if keep_dims:
        dout_expand = dout[1]
        out = op(x)
    else:
        dout_expand = expand(dout[1], onehot_axis)
    if onehot_axis >= len(shape_op(out[0])):
        onehot_axis = -1
    onehot = P.OneHot(onehot_axis)
    type_x = F.dtype(x)
    on_value = F.cast(F.scalar_to_array(1.0), type_x)
    off_value = F.cast(F.scalar_to_array(0.0), type_x)
    dx = dout_expand * onehot(out[0], depth, on_value, off_value)
    return dx


@bprop_getters.register(P.MatMul)
def bprop_matmul(self):
    """Grad definition for `MatMul` operation."""
    ta = self.transpose_a
    tb = self.transpose_b
    mul1 = P.MatMul(transpose_a=(ta and tb),
                    transpose_b=(ta or (not tb)))
    mul2 = P.MatMul(transpose_a=((not ta) or tb),
                    transpose_b=(ta and tb))

    def bprop(x, w, out, dout):
        if ta:
            dx = mul1(w, dout)
        else:
            dx = mul1(dout, w)
        if tb:
            dw = mul2(dout, x)
        else:
            dw = mul2(x, dout)
        return dx, dw

    return bprop


@bprop_getters.register(P.BatchMatMul)
def bprop_batchmatmul(self):
    """Grad definition for `BatchMatMul` operation."""
    ta = self.transpose_a
    tb = self.transpose_b
    mul1 = P.BatchMatMul(transpose_a=(ta and tb),
                         transpose_b=(ta or (not tb)))
    mul2 = P.BatchMatMul(transpose_a=((not ta) or tb),
                         transpose_b=(ta and tb))

    def bprop(x, w, out, dout):
        if ta:
            dx = mul1(w, dout)
        else:
            dx = mul1(dout, w)
        if tb:
            dw = mul2(dout, x)
        else:
            dw = mul2(x, dout)
        return binop_grad_common_with_shift(x, w, dx, dw, 2)

    return bprop


@bprop_getters.register(P.Add)
def get_bprop_add(self):
    """Grad definition for `Add` operation."""

    def bprop(x, y, out, dout):
        return binop_grad_common(x, y, dout, dout)

    return bprop


@bprop_getters.register(P.TensorAdd)
def get_bprop_tensor_add(self):
    """Grad definition for `Add` operation."""

    def bprop(x, y, out, dout):
        return binop_grad_common(x, y, dout, dout)

    return bprop


@bprop_getters.register(P.MatrixInverse)
def get_bprop_matrix_inverse(self):
    """Grad definition for `MatrixInverse` operation."""
    matmul_x1 = nn.MatMul(transpose_x1=True)
    matmul_x2 = nn.MatMul(transpose_x2=True)
    neg = P.Neg()

    def bprop(x, out, dout):
        dx = matmul_x2(dout, out)
        dx = matmul_x1(out, dx)
        dx = neg(dx)
        return (dx,)

    return bprop


@bprop_getters.register(P.Neg)
def get_bprop_neg(self):
    """Grad definition for `Neg` operation."""
    neg_grad = P.Neg()

    def bprop(x, out, dout):
        dx = neg_grad(dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.Sub)
def get_bprop_sub(self):
    """Grad definition for `Sub` operation."""
    neg_func = P.Neg()

    def bprop(x, y, out, dout):
        return binop_grad_common(x, y, dout, neg_func(dout))

    return bprop


@bprop_getters.register(P.Mul)
def get_bprop_mul(self):
    """Grad definition for `Mul` operation."""
    mul_func = P.Mul()

    def bprop(x, y, out, dout):
        bc_dx = mul_func(y, dout)
        bc_dy = mul_func(x, dout)
        return binop_grad_common(x, y, bc_dx, bc_dy)

    return bprop


@bprop_getters.register(P.RealDiv)
def get_bprop_real_div(self):
    """Grad definition for `RealDiv` operation."""
    div_op = P.RealDiv()
    neg = P.Neg()
    mul_op = P.Mul()

    def bprop(x, y, out, dout):
        bc_x = div_op(dout, y)
        bc_y = neg(mul_op(bc_x, out))
        return binop_grad_common(x, y, bc_x, bc_y)

    return bprop


@bprop_getters.register(P.Div)
def get_bprop_div(self):
    """Grad definition for `Div` operation."""
    div_op = P.Div()
    neg = P.Neg()
    mul_op = P.Mul()

    def bprop(x, y, out, dout):
        bc_x = div_op(dout, y)
        bc_y = neg(mul_op(bc_x, out))
        return binop_grad_common(x, y, bc_x, bc_y)

    return bprop


@bprop_getters.register(P.DivNoNan)
def get_bprop_div_no_nan(self):
    """Grad definition for `DivNoNan` operation."""
    div_no_nan_op = P.DivNoNan()
    neg = P.Neg()
    mul_op = P.Mul()

    def bprop(x, y, out, dout):
        bc_x = div_no_nan_op(dout, y)
        bc_y = neg(mul_op(bc_x, out))
        return binop_grad_common(x, y, bc_x, bc_y)

    return bprop


@bprop_getters.register(P.Xdivy)
def get_bprop_xdivy(self):
    """Grad definition for `Xdivy` operation."""
    div_op = P.Xdivy()

    def bprop(x, y, out, dout):
        x_dtype = F.dtype(x)
        not_zero_x = F.cast(F.not_equal(x, F.cast(0.0, x_dtype)), x_dtype)
        bc_x = div_op(not_zero_x, y) * dout
        bc_y = div_op(-x, F.square(y)) * dout
        return binop_grad_common(x, y, bc_x, bc_y)

    return bprop


@bprop_getters.register(P.Floor)
def get_bprop_floor(self):
    """Grad definition for `floor` operation."""
    fill_ = P.Fill()
    shape_ = P.Shape()
    dtype_ = P.DType()

    def bprop(x, out, dout):
        bc_x = fill_(dtype_(x), shape_(x), 0.)
        return (bc_x,)

    return bprop


@bprop_getters.register(P.Ceil)
def get_bprop_ceil(self):
    """Grad definition for `ceil` operation."""
    fill_ = P.Fill()
    shape_ = P.Shape()
    dtype_ = P.DType()

    def bprop(x, out, dout):
        bc_x = fill_(dtype_(x), shape_(x), 0.)
        return (bc_x,)

    return bprop


@bprop_getters.register(P.FloorDiv)
def get_bprop_floordiv(self):
    """Grad definition for `FloorDiv` operation."""

    def bprop(x, y, out, dout):
        return zeros_like(x), zeros_like(y)

    return bprop


@bprop_getters.register(P.FloorMod)
def get_bprop_floormod(self):
    """Grad definition for `FloorMod` operation."""

    def bprop(x, y, out, dout):
        bc_x = dout
        bc_y = -dout * (x // y)
        return binop_grad_common(x, y, bc_x, bc_y)

    return bprop


@bprop_getters.register(P.TruncateDiv)
def get_bprop_truncate_div(self):
    """Grad definition for `TruncateDiv` operation."""

    def bprop(x, y, out, dout):
        return zeros_like(x), zeros_like(y)

    return bprop


@bprop_getters.register(P.TruncateMod)
def get_bprop_truncate_mod(self):
    """Grad definition for `TruncateMod` operation."""
    div_op = P.TruncateDiv()

    def bprop(x, y, out, dout):
        bc_x = dout
        bc_y = -dout * div_op(x, y)
        return binop_grad_common(x, y, bc_x, bc_y)

    return bprop


@bprop_getters.register(P.Mod)
def get_bprop_mod(self):
    """Grad definition for `Mod` operation."""

    def bprop(x, y, out, dout):
        bc_x = dout
        bc_y = -dout * (x // y)
        return binop_grad_common(x, y, bc_x, bc_y)

    return bprop


@bprop_getters.register(P.Square)
def get_bprop_square(self):
    """Grad definition for `Square` operation."""
    mul_func = P.Mul()
    fill_func = P.Fill()
    dtype = P.DType()

    def bprop(x, out, dout):
        temp = mul_func(dout, x)
        dx = mul_func(fill_func(dtype(temp), shape_op(x), 2.0), temp)
        return (dx,)

    return bprop


@bprop_getters.register(P.SquaredDifference)
def get_bprop_squared_difference(self):
    """Grad definition for `SquaredDifference` operation."""
    neg = P.Neg()

    def bprop(x, y, out, dout):
        x_grad = 2 * dout * (x - y)
        bc_x = x_grad
        bc_y = neg(x_grad)
        return binop_grad_common(x, y, bc_x, bc_y)

    return bprop


@bprop_getters.register(P.Xlogy)
def get_bprop_xlogy(self):
    """Grad definition for `Xlogy` operation."""
    log_op = P.Xlogy()
    div_op = P.Xdivy()

    def bprop(x, y, out, dout):
        x_dtype = F.dtype(x)
        not_zero_x = F.cast(F.not_equal(x, F.cast(0.0, x_dtype)), x_dtype)
        bc_x = log_op(not_zero_x, y) * dout
        bc_y = div_op(x, y) * dout
        return binop_grad_common(x, y, bc_x, bc_y)

    return bprop


@bprop_getters.register(P.SquareSumAll)
def get_bprop_square_sum_all(self):
    """Grad definition for `SquareSumAll` operation."""
    mul_func = P.Mul()
    fill_func = P.Fill()
    dtype = P.DType()

    def bprop(x, y, out, dout):
        temp_x = mul_func(dout[0], x)
        temp_y = mul_func(dout[1], y)
        dx = mul_func(fill_func(dtype(temp_x), shape_op(x), 2.0), temp_x)
        dy = mul_func(fill_func(dtype(temp_y), shape_op(y), 2.0), temp_y)
        return (dx, dy)

    return bprop


@bprop_getters.register(P.Sqrt)
def get_bprop_sqrt(self):
    """Grad definition for `Sqrt` operation."""
    sqrt_grad = G.SqrtGrad()

    def bprop(x, out, dout):
        dx = sqrt_grad(out, dout)
        return (dx,)

    return bprop


@bprop_getters.register(G.SqrtGrad)
def get_bprop_sqrt_grad(self):
    """Grad definition for `SqrtGrad` operation."""

    def bprop(y, grad, out, dout):
        gy = dout / y
        dy = -gy * out
        dgrad = 0.5 * gy
        return dy, dgrad

    return bprop


@bprop_getters.register(P.Rsqrt)
def get_bprop_rsqrt(self):
    """Grad definition for `Rsqrt` operation."""
    rsqrt_grad = G.RsqrtGrad()

    def bprop(x, out, dout):
        dx = rsqrt_grad(out, dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.Reciprocal)
def get_bprop_reciprocal(self):
    """Grad definition for `Reciprocal` operation."""
    reciprocal_grad = G.ReciprocalGrad()

    def bprop(x, out, dout):
        dx = reciprocal_grad(out, dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.Log)
def get_bprop_log(self):
    """Grad definition for `Log` operation."""
    reciprocal = P.Reciprocal()

    def bprop(x, out, dout):
        g = reciprocal(x)
        dx = g * dout
        return (dx,)

    return bprop


@bprop_getters.register(P.Log1p)
def get_bprop_log1p(self):
    """Grad definition for `Log1p` operation."""
    reciprocal = P.Reciprocal()

    def bprop(x, out, dout):
        x_1p = x + 1
        g = reciprocal(x_1p)
        dx = g * dout
        return (dx,)

    return bprop


@bprop_getters.register(P.Erf)
def get_bprop_erf(self):
    """Grad definition for `Erf` operation."""
    exp = P.Exp()
    square = P.Square()
    sqrt = P.Sqrt()
    cast = P.Cast()
    dtype = P.DType()
    neg = P.Neg()

    def bprop(x, out, dout):
        half_root_pi = cast(2 / sqrt(F.scalar_to_tensor(np.pi)), dtype(x))
        x_square = square(x)
        dx = dout * half_root_pi * exp(neg(x_square))
        return (dx,)

    return bprop


@bprop_getters.register(P.Erfc)
def get_bprop_erfc(self):
    """Grad definition for `Erfc` operation."""
    exp = P.Exp()
    square = P.Square()
    sqrt = P.Sqrt()
    cast = P.Cast()
    dtype = P.DType()
    neg = P.Neg()

    def bprop(x, out, dout):
        half_root_pi = cast(2 / sqrt(F.scalar_to_tensor(np.pi)), dtype(x))
        x_square = square(x)
        dx = dout * (neg(half_root_pi) * exp(neg(x_square)))
        return (dx,)

    return bprop


@bprop_getters.register(P.Pow)
def get_bprop_pow(self):
    """Grad definition for `Pow` operation."""
    pow_op = P.Pow()
    ln = P.Log()

    def bprop(x, power, out, dout):
        bc_dx = power * pow_op(x, power - 1.0) * dout
        x = F.select(x < 0, F.fill(F.dtype(x), F.shape(x), 1), x)
        bc_dpower = out * ln(x) * dout
        return binop_grad_common(x, power, bc_dx, bc_dpower)

    return bprop


@bprop_getters.register(P.Exp)
def get_bprop_exp(self):
    """Grad definition for `Exp` operation."""
    exp_ = P.Exp()

    def bprop(x, out, dout):
        g = exp_(x)
        dx = g * dout
        return (dx,)

    return bprop


@bprop_getters.register(P.Einsum)
def get_bprop_einsum(self):
    """Grad definition for `Einsum` operation."""
    grad_func = G.EinsumGrad(self.equation)

    def bprop(x, out, dout):
        dx = grad_func(x, dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.Expm1)
def get_bprop_expm1(self):
    """Grad definition for `Expm1` operation."""
    exp_ = P.Exp()

    def bprop(x, out, dout):
        g = exp_(x)
        dx = g * dout
        return (dx,)

    return bprop


@bprop_getters.register(P.Minimum)
def get_bprop_minimum(self):
    """Grad definition for `Minimum` operation."""
    input_grad = G.MinimumGrad()

    def bprop(x, y, out, dout):
        dx, dy = input_grad(x, y, dout)
        return dx, dy

    return bprop


@bprop_getters.register(P.Maximum)
def get_bprop_maximum(self):
    """Grad definition for `Maximum` operation."""
    input_grad = G.MaximumGrad()

    def bprop(x, y, out, dout):
        dx, dy = input_grad(x, y, dout)
        return dx, dy

    return bprop


@bprop_getters.register(P.ReduceSum)
def get_bprop_reducesum(self):
    """Grad definition for `ReduceSum` operation."""

    def bprop(x, axis, out, dout):
        dx = _sum_grad(x, axis, dout)
        return dx, zeros_like(axis)

    return bprop


@bprop_getters.register(P.CumSum)
def get_bprop_cumsum(self):
    """Grad definition for `CumSum` operation."""
    cumsum = P.CumSum(exclusive=self.exclusive, reverse=not self.reverse)

    def bprop(x, axis, out, dout):
        return cumsum(dout, axis), zeros_like(axis)

    return bprop


@constexpr
def _split_shape_index(input_shape, axis):
    """Calculate reduce_prod grad transpose indices and perm shape."""
    rank = len(input_shape)
    if isinstance(axis, int):
        axis = tuple([axis])
    reduction_indices = tuple([(i + rank) % rank for i in axis])
    other_indices = tuple(set(range(rank)) - set(reduction_indices))
    reduced_num = reduce(lambda x, y: x * y, [1] + [input_shape[i] for i in reduction_indices])
    other_num = reduce(lambda x, y: x * y, [1] + [input_shape[i] for i in other_indices])
    perm = reduction_indices + other_indices
    return tuple([reduced_num, other_num]), perm


@constexpr
def _invert_permutation(perm):
    """Calculate invert permutation."""
    out = [0] * len(perm)
    for i, value in enumerate(perm):
        out[value] = i
    return tuple(out)


@bprop_getters.register(P.ReduceProd)
def get_bprop_reduceprod(self):
    """Grad definition for `ReduceProd` operation."""
    transpose = P.Transpose()
    left_cumprod = P.CumProd(exclusive=True)
    right_cumprod = P.CumProd(exclusive=True, reverse=True)

    def bprop(x, axis, out, dout):
        """Grad definition for `Product` operation."""
        # Expand dout to full input shape
        input_shape = shape_op(x)
        output_shape_kept_dims = reduced_shape(input_shape, axis)
        dout = reshape(dout, output_shape_kept_dims)
        tile_scaling = tuple_div(input_shape, output_shape_kept_dims)
        grad = tile(dout, tile_scaling)

        # Pack all reduced dimensions into a single one, so we can perform the cumprod ops.
        pack_shape, perm = _split_shape_index(input_shape, axis)
        permuted = transpose(x, perm)
        permuted_shape = shape_op(permuted)
        reshaped = reshape(permuted, pack_shape)

        # Calculate product, leaving out the current entry
        left = left_cumprod(reshaped, 0)
        right = right_cumprod(reshaped, 0)
        y = reshape(left * right, permuted_shape)

        # Invert the transpose and reshape operations.
        # Make sure to set the statically known shape information through a reshape.
        out = transpose(y, _invert_permutation(perm)) * grad
        dx = reshape(out, input_shape)
        return dx, zeros_like(axis)

    return bprop


@bprop_getters.register(P.CumProd)
def get_bprop_cumprod(self):
    """Grad definition for `CumProd` operation."""
    cumprod = P.CumProd(exclusive=self.exclusive, reverse=self.reverse)
    cumsum = P.CumSum(exclusive=self.exclusive, reverse=not self.reverse)

    def bprop(x, axis, out, dout):
        """Grad definition for `Product` operation."""
        # This will fails when x contains 0
        prod = cumprod(x, axis)
        out = cumsum(prod * dout, axis)
        return out / x, zeros_like(axis)

    return bprop


@bprop_getters.register(P.ReduceAll)
def get_bprop_reduceall(self):
    """Grad definition for `ReduceAll` operation."""

    def bprop(x, axis, out, dout):
        return zeros_like(x), zeros_like(axis)

    return bprop


@bprop_getters.register(P.ReduceAny)
def get_bprop_reduceany(self):
    """Grad definition for `ReduceAny` operation."""

    def bprop(x, axis, out, dout):
        return zeros_like(x), zeros_like(axis)

    return bprop


@bprop_getters.register(P.ReduceMax)
def get_bprop_reducemax(self):
    """Grad definition for `Max` operation."""

    def bprop(x, axis, out, dout):
        dx = _min_or_max_grad(x, axis, out, dout)
        return (dx, zeros_like(axis))

    return bprop


@bprop_getters.register(P.ArgMaxWithValue)
def get_bprop_argmaxwithvalue(self):
    """Grad definition for `ArgMaxWithValue` operation."""
    axis = self.axis
    keep_dims = self.keep_dims
    op = P.ArgMaxWithValue(axis)

    def bprop(x, out, dout):
        dx = _argmin_or_argmax_grad(x, axis, keep_dims, op, out, dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.ReduceMin)
def get_bprop_reducemin(self):
    """Grad definition for `ReduceMin` operation."""

    def bprop(x, axis, out, dout):
        dx = _min_or_max_grad(x, axis, out, dout)
        return (dx, zeros_like(axis))

    return bprop


@bprop_getters.register(P.ArgMinWithValue)
def get_bprop_argminwithvalue(self):
    """Generate bprop for ArgMinWithValue"""
    axis = self.axis
    keep_dims = self.keep_dims
    op = P.ArgMinWithValue(axis)

    def bprop(x, out, dout):
        dx = _argmin_or_argmax_grad(x, axis, keep_dims, op, out, dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.ReduceMean)
def get_bprop_reduce_mean(self):
    """Grad definition for `ReduceMean` operation."""
    div_op = P.RealDiv()
    cast = P.Cast()
    dtype = P.DType()

    def bprop(x, axis, out, dout):
        grad = _sum_grad(x, axis, dout)
        shape_x = shape_op(x)
        shape_out = shape_op(out)
        if -1 in shape_x:
            shape_x = dyn_shape_op(x)
            shape_out = dyn_shape_op(out)
            div_shape = reduce_prod(shape_x) / reduce_prod(shape_out)
            dx = div_op(grad, cast(div_shape, dtype(grad)))
        else:
            div_shape = F.shape_mul(shape_x) / F.shape_mul(shape_out)
            dx = div_op(grad, cast(F.scalar_to_array(div_shape), dtype(grad)))
        return dx, zeros_like(axis)

    return bprop


@bprop_getters.register(P.IsFinite)
def get_bprop_isfinite(self):
    """Grad definition for `IsFinite` operation."""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(P.IsNan)
def get_bprop_isnan(self):
    """Grad definition for `IsNan` operation."""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(P.IsInf)
def get_bprop_isinf(self):
    """Grad definition for `IsInf` operation."""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(P.Equal)
def get_bprop_equal(self):
    """Grad definition for `Equal` operation."""

    def bprop(x, y, out, dout):
        return zeros_like(x), zeros_like(y)

    return bprop


@bprop_getters.register(P.NotEqual)
def get_bprop_not_equal(self):
    """Grad definition for `NotEqual` operation."""

    def bprop(x, y, out, dout):
        return zeros_like(x), zeros_like(y)

    return bprop


@bprop_getters.register(P.ApproximateEqual)
def get_bprop_approximate_equal(self):
    """Grad definition for `ApproximateEqual` operation."""

    def bprop(x, y, out, dout):
        return zeros_like(x), zeros_like(y)

    return bprop


@bprop_getters.register(P.Greater)
def get_bprop_greater(self):
    """Grad definition for `Greater` operation."""

    def bprop(x, y, out, dout):
        return zeros_like(x), zeros_like(y)

    return bprop


@bprop_getters.register(P.GreaterEqual)
def get_bprop_greater_equal(self):
    """Grad definition for `GreaterEqual` operation."""

    def bprop(x, y, out, dout):
        return zeros_like(x), zeros_like(y)

    return bprop


@bprop_getters.register(P.Less)
def get_bprop_less(self):
    """Grad definition for `Less` operation."""

    def bprop(x, y, out, dout):
        return zeros_like(x), zeros_like(y)

    return bprop


@bprop_getters.register(P.LessEqual)
def get_bprop_less_equal(self):
    """Grad definition for `LessEqual` operation."""

    def bprop(x, y, out, dout):
        return zeros_like(x), zeros_like(y)

    return bprop


@bprop_getters.register(P.LogicalNot)
def get_bprop_logical_not(self):
    """Grad definition for `LogicalNot` operation."""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(P.LogicalAnd)
def get_bprop_logical_and(self):
    """Grad definition for `LogicalAnd` operation."""

    def bprop(x, y, out, dout):
        return zeros_like(x), zeros_like(y)

    return bprop


@bprop_getters.register(P.LogicalOr)
def get_bprop_logical_or(self):
    """Grad definition for `LogicalOr` operation."""

    def bprop(x, y, out, dout):
        return zeros_like(x), zeros_like(y)

    return bprop


@bprop_getters.register(P.NPUAllocFloatStatus)
def get_bprop_npu_alloc_float_status(self):
    """Grad definition for `NPUAllocFloatStatus` operation."""

    def bprop(out, dout):
        return ()

    return bprop


@bprop_getters.register(P.NPUGetFloatStatus)
def get_bprop_npu_get_float_status(self):
    """Grad definition for `NPUGetFloatStatus` operation."""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(P.NPUClearFloatStatus)
def get_bprop_npu_clear_float_status(self):
    """Grad definition for `NPUClearFloatStatus` operation."""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(P.AssignAdd)
def get_bprop_assign_add(self):
    """Grad definition for `AssignAdd` operation."""

    def bprop(x, y, out, dout):
        return zeros_like(x), zeros_like(y)

    return bprop


@bprop_getters.register(P.AssignSub)
def get_bprop_assign_sub(self):
    """Grad definition for `AssignSub` operation."""

    def bprop(x, y, out, dout):
        return zeros_like(x), zeros_like(y)

    return bprop


@bprop_getters.register(P.Sin)
def get_bprop_sin(self):
    """Grad definition for `Sin` operation."""
    cos = P.Cos()

    def bprop(x, out, dout):
        dx = dout * cos(x)
        return (dx,)

    return bprop


@bprop_getters.register(P.Asin)
def get_bprop_asin(self):
    """Grad definition for `Asin` operation."""
    input_grad = G.AsinGrad()

    def bprop(x, out, dout):
        dx = input_grad(x, dout)
        return (dx,)

    return bprop


@bprop_getters.register(G.AsinGrad)
def get_bprop_asin_grad(self):
    """Grad definition for `AsinGrad` operation."""
    input_grad = G.AsinGrad()
    p_pow = P.Pow()

    def bprop(x, grad, out, dout):
        d2x = dout * grad * x * p_pow((1 - x * x), - 1.5)
        ddy = input_grad(x, dout)
        return (d2x, ddy)

    return bprop


@bprop_getters.register(P.Asinh)
def get_bprop_asinh(self):
    """Grad definition for `Asinh` operation."""
    input_grad = G.AsinhGrad()

    def bprop(x, out, dout):
        dx = input_grad(out, dout)
        return (dx,)

    return bprop


@bprop_getters.register(G.AsinhGrad)
def get_bprop_asinh_grad(self):
    """Grad definition for `AsinhGrad` operation."""
    input_grad = G.AsinhGrad()
    tanh = P.Tanh()

    def bprop(y, grad, out, dout):
        dy = dout * out * -1.0 * tanh(y)
        dgrad = input_grad(y, dout)
        return dy, dgrad

    return bprop


@bprop_getters.register(P.Sinh)
def get_bprop_sinh(self):
    """Grad definition for `Sinh` operation."""
    cosh = P.Cosh()

    def bprop(x, out, dout):
        dx = cosh(x) * dout
        return (dx,)

    return bprop


@bprop_getters.register(P.Cos)
def get_bprop_cos(self):
    """Grad definition for `Cos` operation."""
    sin = P.Sin()
    neg = P.Neg()

    def bprop(x, out, dout):
        dx = dout * neg(sin(x))
        return (dx,)

    return bprop


@bprop_getters.register(P.ACos)
def get_bprop_acos(self):
    """Grad definition for `ACos` operation."""
    input_grad = G.ACosGrad()

    def bprop(x, out, dout):
        dx = input_grad(x, dout)
        return (dx,)

    return bprop


@bprop_getters.register(G.ACosGrad)
def get_bprop_acos_grad(self):
    """Grad definition for `ACosGrad` operation."""
    input_grad = G.ACosGrad()
    p_pow = P.Pow()

    def bprop(x, grad, out, dout):
        d2x = -dout * grad * x * p_pow((1 - x * x), - 1.5)
        ddy = input_grad(x, dout)
        return (d2x, ddy)

    return bprop


@bprop_getters.register(P.Acosh)
def get_bprop_acosh(self):
    """Grad definition for `Acosh` operation."""
    input_grad = G.AcoshGrad()

    def bprop(x, out, dout):
        dx = input_grad(out, dout)
        return (dx,)

    return bprop


@bprop_getters.register(G.AcoshGrad)
def get_bprop_acosh_grad(self):
    """Grad definition for `AcoshGrad` operation."""
    input_grad = G.AcoshGrad()
    tanh = P.Tanh()

    def bprop(y, grad, out, dout):
        dy = dout * out * -1.0 / tanh(y)
        dgrad = input_grad(y, dout)
        return dy, dgrad

    return bprop


@bprop_getters.register(P.Cosh)
def get_bprop_cosh(self):
    """Grad definition for `Cosh` operation."""
    sinh = P.Sinh()

    def bprop(x, out, dout):
        dx = sinh(x) * dout
        return (dx,)

    return bprop


@bprop_getters.register(P.Abs)
def get_bprop_abs(self):
    """Grad definition for `Abs` operation."""
    abs_grad = G.AbsGrad()

    def bprop(x, out, dout):
        dx = abs_grad(x, dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.Conj)
def get_bprop_conj(self):
    """Grad definition for `Conj` operation."""
    conj = P.Conj()

    def bprop(x, out, dout):
        dx = conj(dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.Real)
def get_bprop_real(self):
    """Grad definition for `Real` operation."""
    cast = P.Cast()
    dtype = P.DType()

    def bprop(x, out, dout):
        return (cast(dout, dtype(x)),)

    return bprop


@bprop_getters.register(P.Imag)
def get_bprop_imag(self):
    """Grad definition for `Imag` operation."""
    complex_op = P.Complex()

    def bprop(x, out, dout):
        zeros = zeros_like(dout)
        return (complex_op(zeros, zeros - 1) * dout,)

    return bprop


@bprop_getters.register(P.ScalarCast)
def get_bprop_scalar_cast(self):
    """Generate bprop for ScalarCast"""

    def bprop(x, t, out, dout):
        return F.scalar_cast(dout, F.typeof(x)), zeros_like(t)

    return bprop


@bprop_getters.register(P.AccumulateNV2)
def get_bprop_scalar_accumulatenv2(self):
    """Generate bprop for AccumulateNV2"""

    def bprop(x, out, dout):
        dx = ()
        for _ in range(len(x)):
            dx = dx + (dout,)
        return (dx,)

    return bprop


@bprop_getters.register(P.AddN)
def get_bprop_scalar_addn(self):
    """Generate bprop for AddN"""

    def bprop(x, out, dout):
        if is_sub_class(F.typeof(x), ms.list_):
            dx = []
            for _ in range(len(x)):
                dx.append(dout)
            return (dx,)

        dx = ()
        for _ in range(len(x)):
            dx = dx + (dout,)
        return (dx,)

    return bprop


@bprop_getters.register(P.Sign)
def get_bprop_sign(self):
    """Generate bprop for Sign"""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(P.Round)
def get_bprop_round(self):
    """Generate bprop for Round"""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(P.Atan2)
def get_bprop_atan2(self):
    """Generate bprop for Atan2"""

    square = P.Square()

    def bprop(x, y, out, dout):
        tmp = dout / (square(x) + square(y))
        bc_dx = tmp * y
        bc_dy = tmp * (-x)
        return binop_grad_common(x, y, bc_dx, bc_dy)

    return bprop


@bprop_getters.register(P.BesselI0e)
def get_bprop_bessel_i0e(self):
    """Generate bprop for BesselI0e"""
    sign = P.Sign()
    bessel_i1e = P.BesselI1e()

    def bprop(x, out, dout):
        dx = dout * (bessel_i1e(x) - sign(x) * out)
        return (dx,)

    return bprop


@bprop_getters.register(P.Atan)
def get_bprop_atan(self):
    """Grad definition for `Atan` operation."""
    input_grad = G.AtanGrad()

    def bprop(x, out, dout):
        dx = input_grad(x, dout)
        return (dx,)

    return bprop


@bprop_getters.register(G.AtanGrad)
def get_bprop_atan_grad(self):
    """Grad definition for `AtanGrad` operation."""
    input_grad = G.AtanGrad()

    def bprop(x, grad, out, dout):
        dgrad = input_grad(x, dout)
        dx = out * dgrad * -2.0 * x
        return dx, dgrad

    return bprop


@bprop_getters.register(P.Tan)
def get_bprop_tan(self):
    """Grad definition for `Tan` operation."""
    reciprocal = P.Reciprocal()
    square = P.Square()
    cos = P.Cos()

    def bprop(x, out, dout):
        cosx = cos(x)
        secx2 = square(reciprocal(cosx))
        dx = secx2 * dout
        return (dx,)

    return bprop


@bprop_getters.register(P.BesselI1e)
def get_bprop_bessel_i1e(self):
    """Generate bprop for BesselI1e"""

    sign = P.Sign()
    bessel_i0e = P.BesselI0e()
    less = P.Less()
    select = P.Select()
    reciprocal = P.Reciprocal()
    cast = P.Cast()
    dtype = P.DType()
    abs_ops = P.Abs()

    def bprop(x, out, dout):
        zeros = zeros_like(x)
        np_eps = const_utils.get_np_eps(dtype(x))
        eps = cast(np_eps, dtype(x))
        x_is_valid = less(eps, abs_ops(x))
        x_safe = select(x_is_valid, x, eps + zeros)
        tmp = bessel_i0e(x_safe) - out * (sign(x_safe) + reciprocal(x_safe))
        dx = select(x_is_valid, tmp, cast(0.5, dtype(x)) + zeros) * dout
        return (dx,)

    return bprop


@bprop_getters.register(P.Atanh)
def get_bprop_atanh(self):
    """Grad definition for `Atanh` operation."""
    power = P.Pow()
    div = P.Div()

    def bprop(x, out, dout):
        tmp = 1 - power(x, 2)
        dx = div(1, tmp) * dout
        return (dx,)

    return bprop


@bprop_getters.register(P.Inv)
def get_bprop_inv(self):
    """Grad definition for 'Inv' operation"""
    inv_grad = G.InvGrad()

    def bprop(x, out, dout):
        dx = inv_grad(out, dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.LinSpace)
def get_bprop_lin_space(self):
    """Grad definition for `LinSpace` operation."""

    def bprop(start, stop, num, out, dout):
        return zeros_like(start), zeros_like(stop), zeros_like(num)

    return bprop


@bprop_getters.register(P.IndexAdd)
def get_bprop_index_add(self):
    """Generate bprop for IndexAdd"""
    gather = P.Gather()
    _axis = self.axis

    def bprop(input_x, indices, input_y, out, dout):
        return dout, zeros_like(indices), gather(dout, indices, _axis)

    return bprop
