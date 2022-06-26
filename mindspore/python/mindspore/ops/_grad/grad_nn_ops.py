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

"""Define the grad rules of neural network related operations."""
from mindspore.ops.primitive import constexpr
from mindspore.ops.operations import nn_ops as nps
from .grad_base import bprop_getters
from .. import functional as F
from .. import operations as P
from ...common import dtype as mstype
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from ..operations import _grad_ops as G
from ..operations import _inner_ops as inner
from ..operations import _rl_inner_ops as rl_ops
from ... import context
from .._utils.utils import range_op, get_1d_shape


@bprop_getters.register(P.BiasAdd)
def get_bprop_bias_add(self):
    """Grad definition for `BiasAdd` operation."""
    bias_grad = G.BiasAddGrad(self.data_format)

    def bprop(x, w, out, dout):
        return dout, bias_grad(dout)

    return bprop


@bprop_getters.register(P.Conv2D)
def get_bprop_conv2d(self):
    """Grad definition for `Conv2D` operation."""
    self.out_channel = self.get_attr_dict()["out_channel"]
    self.pad_list = self.get_attr_dict()["pad_list"]
    input_grad = P.Conv2DBackpropInput(
        self.out_channel, self.kernel_size, self.pad_mode, self.pad, self.pad_list, mode=self.mode,
        dilation=self.dilation, stride=self.stride, group=self.group, data_format=self.format
    )
    filter_grad = G.Conv2DBackpropFilter(
        self.out_channel, self.kernel_size, self.pad_mode, self.pad, self.pad_list, mode=self.mode,
        dilation=self.dilation, stride=self.stride, group=self.group, data_format=self.format
    )
    get_shape = P.Shape()
    get_dyn_shape = P.TensorShape()

    def bprop(x, w, out, dout):
        x_shape = get_shape(x)
        w_shape = get_shape(w)
        if -1 in x_shape:
            x_shape = get_dyn_shape(x)
        if -1 in w_shape:
            w_shape = get_dyn_shape(w)
        dx = input_grad(dout, w, x_shape)

        dw = filter_grad(dout, x, w_shape)
        return dx, dw

    return bprop


@bprop_getters.register(nps.Conv3D)
def get_bprop_conv3d(self):
    """Grad definition for `Conv3D` operation."""
    input_grad = nps.Conv3DBackpropInput(
        self.out_channel, self.kernel_size, self.mode, pad_mode=self.pad_mode,
        pad=self.pad, stride=self.stride, dilation=self.dilation, group=self.group, data_format=self.data_format
    )
    filter_grad = G.Conv3DBackpropFilter(
        self.out_channel, self.kernel_size, self.mode, pad_mode=self.pad_mode,
        pad=self.pad, stride=self.stride, dilation=self.dilation, group=self.group, data_format=self.data_format
    )
    get_shape = P.Shape()

    def bprop(x, w, out, dout):
        dx = input_grad(w, dout, get_shape(x))
        dw = filter_grad(x, dout, get_shape(w))
        return dx, dw

    return bprop


@bprop_getters.register(nps.Conv3DTranspose)
def get_bprop_conv3d_transpose(self):
    """Grad definition for `Conv3DTranspose` operation."""
    stride = (self.stride[2], self.stride[3], self.stride[4])
    dilation = (self.dilation[2], self.dilation[3], self.dilation[4])
    input_grad = nps.Conv3D(
        out_channel=self.in_channel, kernel_size=self.kernel_size, mode=self.mode, pad_mode="pad",
        pad=self.pad_list, stride=stride, dilation=dilation, group=self.group, data_format=self.data_format
    )
    filter_grad = G.Conv3DBackpropFilter(
        out_channel=self.in_channel, kernel_size=self.kernel_size, mode=self.mode, pad_mode="pad",
        pad=self.pad_list, stride=self.stride, dilation=self.dilation, group=self.group, data_format=self.data_format
    )

    def bprop(x, w, out, dout):
        dx = input_grad(dout, w)
        dw = filter_grad(dout, x, F.shape(w))
        return dx, dw

    return bprop


@bprop_getters.register(inner.ExtractImagePatches)
def get_bprop_extract_image_patches(self):
    """Grad definition for `ExtractImagePatches` operation."""
    get_shape = P.Shape()
    reshape = P.Reshape()
    extract_image_patches = inner.ExtractImagePatches(ksizes=self.ksizes,
                                                      strides=self.strides,
                                                      rates=self.rates,
                                                      padding=self.padding)
    concat = P.Concat(axis=-1)
    expand_dims = P.ExpandDims()
    scatter_nd = P.ScatterNd()
    dtype = P.DType()
    fill = P.Fill()
    slice_op = P.Slice()
    transpose = P.Transpose()
    cast = P.Cast()
    matmul = P.MatMul()

    _, _, ksizes_row, ksizes_col = self.ksizes

    def bprop(x, out, dout):
        x_shape = get_shape(x)
        x_batch, x_depth, x_row, x_col = x_shape
        x_indices_num = x_row * x_col + 1
        x_idx = cast(F.tuple_to_array(range(1, x_indices_num)), mstype.float32)
        x_idx = reshape(x_idx, (1, 1, x_row, x_col))
        x_idx_patch = cast(extract_image_patches(x_idx), mstype.int32)
        x_idx_patch = transpose(x_idx_patch, (0, 2, 3, 1))

        out_shape = get_shape(out)
        _, _, out_row, out_col = out_shape
        out_indices_num = out_row * out_col * ksizes_row * ksizes_col
        out_idx = F.tuple_to_array(range(out_indices_num))
        out_idx = reshape(out_idx, (1, out_row, out_col, ksizes_row * ksizes_col))

        idx_tensor = concat((expand_dims(x_idx_patch, -1), expand_dims(out_idx, -1)))
        idx_tensor = reshape(idx_tensor, (-1, 2))
        sp_shape = (x_indices_num, out_indices_num)
        sp_tensor = scatter_nd(idx_tensor, fill(dtype(dout), (out_indices_num,), 1), sp_shape)
        sp_tensor = slice_op(sp_tensor, (1, 0), (x_indices_num - 1, out_indices_num))

        grad = transpose(dout, (0, 2, 3, 1))
        grad = reshape(grad, (x_batch, out_row, out_col, ksizes_row, ksizes_col, x_depth))
        grad = transpose(grad, (1, 2, 3, 4, 0, 5))
        grad = reshape(grad, (-1, x_batch * x_depth))

        jac = matmul(sp_tensor, grad)
        dx = reshape(jac, (x_row, x_col, x_batch, x_depth))
        dx = transpose(dx, (2, 3, 0, 1))
        return (dx,)

    return bprop


@bprop_getters.register(P.DepthwiseConv2dNative)
def get_bprop_depthwise_conv2d_native(self):
    """Grad definition for `DepthwiseConv2dNative` operation."""
    input_grad = G.DepthwiseConv2dNativeBackpropInput(
        self.channel_multiplier, self.kernel_size, self.pad_mode, self.pad, self.pad_list, self.mode, self.stride,
        self.dilation, self.group
    )
    filter_grad = G.DepthwiseConv2dNativeBackpropFilter(
        self.channel_multiplier, self.kernel_size, self.pad_mode, self.pad, self.pad_list, self.mode, self.stride,
        self.dilation, self.group
    )
    get_shape = P.Shape()

    def bprop(x, w, out, dout):
        dx = input_grad(get_shape(x), w, dout)

        dw = filter_grad(x, get_shape(w), dout)
        return dx, dw

    return bprop


@bprop_getters.register(P.MaxPoolWithArgmax)
def get_bprop_max_pool_with_argmax(self):
    """Grad definition for `MaxPoolWithArgmax` operation."""
    maxpool_grad = G.MaxPoolGradWithArgmax(
        kernel_size=self.kernel_size,
        strides=self.strides,
        pad_mode=self.pad_mode)

    def bprop(x, out, dout):
        dx = maxpool_grad(x, dout[0], out[1])
        return (dx,)

    return bprop


@bprop_getters.register(G.MaxPoolGrad)
def get_bprop_max_pool_grad_grad(self):
    """Grad definition for `MaxPoolGrad` operation."""
    device_target = context.get_context("device_target")
    is_ascend = (device_target == "Ascend")
    if device_target == "Ascend":
        maxpool_grad_grad = G.MaxPoolGradGrad(
            kernel_size=self.kernel_size,
            strides=self.strides,
            pad_mode=self.pad_mode)
    elif device_target == "GPU":
        if self.data_format != "NCHW":
            raise RuntimeError("MaxPoolGradGrad does not support NHWC!")
        kernel_size = self.kernel_size
        if isinstance(kernel_size, tuple) and len(kernel_size) == 4:
            kernel_size = kernel_size[2:]
        strides = self.strides
        if isinstance(strides, tuple) and len(strides) == 4:
            strides = strides[2:]
        maxpool_with_argmax = P.MaxPoolWithArgmax(kernel_size=kernel_size, strides=strides, pad_mode=self.pad_mode)
        gather = P.GatherNd()
        reshape = P.Reshape()
    else:
        raise RuntimeError("MaxPoolGradGrad does not support on CPU!")

    def bprop(x1, x2, grad, out, dout):
        dx1 = zeros_like(x1)
        dx2 = zeros_like(x2)
        if is_ascend:
            dgrad = maxpool_grad_grad(x1, x2, dout)
        else:
            b, c, h, w = P.Shape()(x2)
            _, ind = maxpool_with_argmax(x1)
            batch = F.cast(F.tuple_to_array(range(b)), mstype.int32)
            batch = P.Tile()(reshape(batch, (-1, 1)), (1, c * h * w))
            gather_ind = P.Stack(-1)((batch, reshape(ind, (b, -1))))
            dgrad = reshape(gather(reshape(dout, (b, -1)), gather_ind), (b, c, h, w))
        return (dx1, dx2, dgrad)

    return bprop


@bprop_getters.register(G.MaxPoolGradGrad)
def get_bprop_max_pool_grad_grad_grad(self):
    """Grad definition for `MaxPoolGradGrad` operation."""
    maxpool_grad = G.MaxPoolGrad(
        kernel_size=self.kernel_size,
        strides=self.strides,
        pad_mode=self.pad_mode)

    def bprop(x1, x2, grad, out, dout):
        dx1 = zeros_like(x1)
        dx2 = zeros_like(x2)
        dgrad = maxpool_grad(x1, x2, dout)
        return (dx1, dx2, dgrad)

    return bprop


@bprop_getters.register(P.MaxPool)
def get_bprop_max_pool_grad(self):
    """Grad definition for `MaxPool` operation."""
    maxpool_grad = G.MaxPoolGrad(
        kernel_size=self.kernel_size,
        strides=self.strides,
        pad_mode=self.pad_mode,
        data_format=self.format)

    def bprop(x, out, dout):
        dx = maxpool_grad(x, out, dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.MaxPool3D)
def get_bprop_max_pool3d_grad(self):
    """Grad definition for `MaxPool3D` operation."""
    max_pool3d_grad = G.MaxPool3DGrad(
        kernel_size=self.kernel_size,
        strides=self.strides,
        pad_mode=self.pad_mode,
        pad_list=self.pad_list,
        data_format=self.data_format)

    def bprop(x, out, dout):
        dx = max_pool3d_grad(x, out, dout)
        return (dx,)

    return bprop


@bprop_getters.register(G.MaxPool3DGrad)
def get_bprop_max_pool3d_grad_grad(self):
    """Grad definition for `MaxPool3Grad` operation."""
    max_pool3d_grad_grad = G.MaxPool3DGradGrad(
        kernel_size=self.kernel_size,
        strides=self.strides,
        pad_mode=self.pad_mode,
        data_format=self.data_format)

    def bprop(x, y, grad, out, dout):
        dgrad = max_pool3d_grad_grad(x, y, dout)
        return zeros_like(x), zeros_like(y), dgrad

    return bprop


@bprop_getters.register(G.MaxPool3DGradGrad)
def get_bprop_max_pool3d_grad_grad_grad(self):
    """Grad definition for `MaxPool3GradGrad` operation."""

    max_pool3d_grad = G.MaxPool3DGrad(
        kernel_size=self.kernel_size,
        strides=self.strides,
        pad_mode=self.pad_mode,
        data_format=self.data_format)

    def bprop(x, y, grad, out, dout):
        dgrad = max_pool3d_grad(x, y, dout)
        return zeros_like(x), zeros_like(y), dgrad

    return bprop


@bprop_getters.register(P.AvgPool)
def get_bprop_avg_pool_grad(self):
    """Grad definition for `AvgPool` operation."""
    avgpool_grad = G.AvgPoolGrad(
        kernel_size=self.kernel_size,
        strides=self.strides,
        pad_mode=self.pad_mode,
        data_format=self.format)

    def bprop(x, out, dout):
        dx = avgpool_grad(x, out, dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.AdaptiveAvgPool2D)
def get_bprop_adaptive_avg_pool2d_grad(self):
    """Grad definition for `AdaptiveAvgPool2D` operation."""
    adaptive_avgpool_grad = G.AdaptiveAvgPool2DGrad()

    def bprop(x, out, dout):
        dx = adaptive_avgpool_grad(x, dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.AvgPool3D)
def get_bprop_avg_pool_3d_grad(self):
    """Grad definition for `AvgPool3D` operation."""
    pad_list = self.get_attr_dict()['pad_list']
    count_include_pad = self.get_attr_dict()['count_include_pad']
    avgpool3d_grad = G.AvgPool3DGrad(kernel_size=self.kernel_size,
                                     strides=self.strides,
                                     pads=pad_list,
                                     ceil_mode=self.ceil_mode,
                                     count_include_pad=count_include_pad,
                                     divisor_override=self.divisor_override,
                                     data_format=self.data_format,
                                     pad_mode=self.pad_mode)

    def bprop(x, out, dout):
        x_shape = F.shape(x)
        dx = avgpool3d_grad(x_shape, dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.DropoutGenMask)
def get_bprop_dropout_gen_mask(self):
    """Grad definition for `DropoutGenMask` operation."""

    def bprop(shape, keep_prob, out, dout):
        return (zeros_like(shape), zeros_like(keep_prob))

    return bprop


@bprop_getters.register(P.DropoutDoMask)
def get_bprop_dropout_do_mask(self):
    """Grad definition for `DropoutDoMask` operation."""
    do_mask = P.DropoutDoMask()

    def bprop(x, y, keep_prob, out, dout):
        return (do_mask(dout, y, keep_prob), zeros_like(y), zeros_like(keep_prob))

    return bprop


@bprop_getters.register(P.Mish)
def get_bprop_mish(self):
    """Grad definition for `Mish` operation."""
    tanh = P.Tanh()
    tanh_grad = G.TanhGrad()
    softplus = P.Softplus()
    softplus_grad = G.SoftplusGrad()

    def bprop(x, out, dout):
        dx1 = tanh(softplus(x))
        dx2 = softplus_grad(tanh_grad(dx1, x * dout), x)
        dx = (dx1 * dout + dx2)
        return (dx,)

    return bprop


@bprop_getters.register(P.SeLU)
def get_bprop_selu(self):
    """Grad definition for `SeLU` operation."""
    scale = 1.0507009873554804934193349852946
    elu_grad = G.EluGrad()

    def bprop(x, out, dout):
        dx = elu_grad(dout, out) * scale
        return (dx,)

    return bprop


@bprop_getters.register(P.MulNoNan)
def get_bprop_mul_no_nan(self):
    """Grad definition for `MulNoNan` operation."""
    mul_no_nan = P.MulNoNan()
    reduce_sum = P.ReduceSum()
    reshape = P.Reshape()

    def bprop(x, y, out, dout):
        x_shape = F.shape(x)
        y_shape = F.shape(y)
        dx = mul_no_nan(dout, y)
        dy = mul_no_nan(x, dout)
        broadcast_x, broadcast_y = F.broadcast_gradient_args(x_shape, y_shape)
        if broadcast_x != ():
            dx = reshape(reduce_sum(dx, broadcast_x), x_shape)
        if broadcast_y != ():
            dy = reshape(reduce_sum(dy, broadcast_y), y_shape)
        return dx, dy

    return bprop


@bprop_getters.register(P.ReLU)
def get_bprop_relu(self):
    """Grad definition for `ReLU` operation."""
    input_grad = G.ReluGrad()

    def bprop(x, out, dout):
        dx = input_grad(dout, out)
        return (dx,)

    return bprop


@bprop_getters.register(G.ReluGrad)
def get_bprop_relu_grad(self):
    """Grad definition for `ReLUGrad` operation."""
    input_grad = G.ReluGrad()

    def bprop(grad, y, out, dout):
        dgrad = input_grad(dout, y)
        return dgrad, zeros_like(y)

    return bprop


@bprop_getters.register(P.ReLU6)
def get_bprop_relu6(self):
    """Grad definition for `ReLU6` operation."""
    input_grad = G.ReLU6Grad()

    def bprop(x, out, dout):
        dx = input_grad(dout, x)
        return (dx,)

    return bprop


@bprop_getters.register(P.ReLUV2)
def get_bprop_relu_v2(self):
    """Grad definition for `ReLUV2` operation."""
    input_grad = G.ReluGradV2()

    def bprop(x, out, dout):
        mask = out[1]
        dx = input_grad(dout[0], mask)
        return (dx,)

    return bprop


@bprop_getters.register(P.HSwish)
def get_bprop_hswish(self):
    """Grad definition for `HSwish` operation."""
    input_grad = G.HSwishGrad()

    def bprop(x, out, dout):
        dx = input_grad(dout, x)
        return (dx,)

    return bprop


@bprop_getters.register(P.HSigmoid)
def get_bprop_hsigmoid(self):
    """Grad definition for `HSigmoid` operation."""
    input_grad = G.HSigmoidGrad()

    def bprop(x, out, dout):
        dx = input_grad(dout, x)
        return (dx,)

    return bprop


@bprop_getters.register(P.Elu)
def get_bprop_elu(self):
    """Grad definition for `Elu` operation."""
    input_grad = G.EluGrad()

    def bprop(x, out, dout):
        dx = input_grad(dout, out)
        return (dx,)

    return bprop


@bprop_getters.register(P.Sigmoid)
def get_bprop_sigmoid(self):
    """Grad definition for `Sigmoid` operation."""
    input_grad = G.SigmoidGrad()

    def bprop(x, out, dout):
        dx = input_grad(out, dout)
        return (dx,)

    return bprop


@bprop_getters.register(G.SigmoidGrad)
def get_bprop_sigmoid_grad(self):
    """Grad definition for `SigmoidGrad` operation."""
    sigmoid_grad = G.SigmoidGrad()

    def bprop(y, grad, out, dout):
        dy = dout * grad * (1. - 2 * y)
        dgrad = sigmoid_grad(y, dout)
        return dy, dgrad

    return bprop


@constexpr
def _get_transpose_axis(x_shp, axis):
    rank = len(x_shp)
    if axis < 0:
        axis += rank
    reverse_axis = [i for i in range(rank)]
    reverse_axis[axis] = rank - 1
    reverse_axis[rank - 1] = axis
    return tuple(reverse_axis)


@bprop_getters.register(P.Softmax)
def get_bprop_softmax(self):
    """Grad definition for `Softmax` operation."""
    sum_func = P.ReduceSum(keep_dims=True)
    sub = P.Sub()
    mul = P.Mul()
    get_shape = P.Shape()
    transpose = P.Transpose()
    axis = self.axis
    if not isinstance(axis, int):
        axis = axis[0]

    def bprop(x, out, dout):
        # dx = (dout - sum(dout * out)) * out
        # This formula is correct only when the `axis` is the last dimension.
        # In order to support the scenario where the `axis` is other values,
        # we transpose the data of the `axis` dimension to the last dimension for calculation,
        # and then transpose it back after the calculation.
        reverse_axis = _get_transpose_axis(get_shape(x), axis)
        out = transpose(out, reverse_axis)
        dout = transpose(dout, reverse_axis)
        dx = mul(out, sub(dout, sum_func(mul(out, dout), -1)))
        dx = transpose(dx, reverse_axis)
        return (dx,)

    return bprop


@bprop_getters.register(P.LogSoftmax)
def get_bprop_log_softmax(self):
    """Grad definition for `LogSoftmax` operation."""
    logsoftmax_grad = G.LogSoftmaxGrad(self.axis)

    def bprop(x, out, dout):
        dx = logsoftmax_grad(out, dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.Softplus)
def get_bprop_softplus(self):
    """Grad definition for `Softplus` operation."""
    softplus_grad = G.SoftplusGrad()

    def bprop(x, out, dout):
        dx = softplus_grad(dout, x)
        return (dx,)

    return bprop


@bprop_getters.register(P.Softsign)
def get_bprop_softsign(self):
    """Grad definition for `Softsign` operation."""
    mul = P.Mul()
    absolute = P.Abs()
    div = P.Div()
    square = P.Square()

    def bprop(x, out, dout):
        dx = mul(dout, div(1, square(1 + absolute(x))))
        return (dx,)

    return bprop


@bprop_getters.register(P.Tanh)
def get_bprop_tanh(self):
    """Grad definition for `Tanh` operation."""
    tanh_grad = G.TanhGrad()

    def bprop(x, out, dout):
        dx = tanh_grad(out, dout)
        return (dx,)

    return bprop


@bprop_getters.register(G.TanhGrad)
def get_bprop_tanh_grad(self):
    """Grad definition for `TanhGrad` operation."""
    tanh_grad = G.TanhGrad()

    def bprop(y, grad, out, dout):
        dy = dout * -2.0 * grad * y
        dgrad = tanh_grad(y, dout)
        return dy, dgrad

    return bprop


@bprop_getters.register(P.Gelu)
@bprop_getters.register(P.GeLU)
def get_bprop_gelu(self):
    """Grad definition for `GeLU` operation."""
    input_grad = G.GeLUGrad()

    def bprop(x, out, dout):
        dx = input_grad(dout, x, out)
        return (dx,)

    return bprop


@bprop_getters.register(P.FastGeLU)
def get_bprop_fast_gelu(self):
    """Grad definition for `FastGeLU` operation."""
    input_grad = G.FastGeLUGrad()

    def bprop(x, out, dout):
        dx = input_grad(dout, x)
        return (dx,)

    return bprop


@bprop_getters.register(P.FastGelu)
def get_bprop_fast_gelu_2(self):
    """Grad definition for `FastGeLU` operation."""
    input_grad = G.FastGeLUGrad()

    def bprop(x, out, dout):
        dx = input_grad(dout, x)
        return (dx,)

    return bprop


@bprop_getters.register(P.InstanceNorm)
def get_bprop_instance_norm(self):
    """Grad definition for `InstanceNorm` operation."""
    input_grad = G.InstanceNormGrad(self.epsilon, self.momentum)

    def bprop(x, gamma, beta, mean, variance, out, dout):
        saved_mean = out[1]
        saved_variance = out[2]
        out = input_grad(dout[0], x, gamma, saved_mean, saved_variance)
        dx = out[0]
        dgamma = out[1]
        dbeta = out[2]
        return dx, dgamma, dbeta, zeros_like(mean), zeros_like(variance)

    return bprop


@bprop_getters.register(P.BatchNorm)
def get_bprop_batch_norm(self):
    """Grad definition for `BatchNorm` operation."""
    is_training = self.is_training
    input_grad = G.BatchNormGrad(is_training, self.epsilon, self.data_format)

    def bprop(x, scale, b, mean, variance, out, dout):
        if is_training:
            saved_mean = out[3]
            saved_variance = out[4]
            reserve = out[2]
        else:
            saved_mean = mean
            saved_variance = variance
            reserve = out[2]
        out = input_grad(dout[0], x, scale, saved_mean, saved_variance, reserve)
        dx = out[0]
        dscale = out[1]
        dbias = out[2]
        return dx, dscale, dbias, zeros_like(mean), zeros_like(variance)

    return bprop


@bprop_getters.register(P.LayerNorm)
def get_bprop_layer_norm(self):
    """Grad definition for `LayerNorm` operation."""
    layer_norm_grad = G.LayerNormGrad(self.begin_norm_axis, self.begin_params_axis)

    def bprop(x, gamma, beta, out, dout):
        dx, d_gamma, d_beta = layer_norm_grad(
            x, dout[0], out[2], out[1], gamma)
        return dx, d_gamma, d_beta

    return bprop


@bprop_getters.register(G.LayerNormGrad)
def get_bprop_layer_norm_grad(self):
    """Grad definition for `LayerNormGrad` operation."""
    layer_norm_grad_grad = G.LayerNormGradGrad(self.begin_norm_axis, self.begin_params_axis)

    def bprop(x, dy, variance, mean, gamma, out, dout):
        d_x, d_dy, d_gamma = layer_norm_grad_grad(
            x, dy, variance, mean, gamma, dout[0], dout[1], dout[2])
        return d_x, d_dy, zeros_like(variance), zeros_like(mean), d_gamma

    return bprop


@bprop_getters.register(P.L2Normalize)
def get_bprop_l2normalize(self):
    """Grad definition for `L2Normalize` operation."""
    input_grad = G.L2NormalizeGrad(self.axis, self.epsilon)

    def bprop(x, out, dout):
        dx = input_grad(x, out, dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.SoftmaxCrossEntropyWithLogits)
def get_bprop_softmax_cross_entropy_with_logits(self):
    """Grad definition for `SoftmaxCrossEntropyWithLogits` operation."""
    expand = P.ExpandDims()

    def bprop(logits, labels, out, dout):
        grad = out[1]
        grad = grad * expand(dout[0], -1)
        return grad, zeros_like(labels)

    return bprop


@bprop_getters.register(P.NLLLoss)
def get_bprop_nll_loss(self):
    """Grad definition for `NLLLoss` operation."""
    nll_loss_grad = G.NLLLossGrad(reduction=self.reduction)

    def bprop(x, target, weight, out, dout):
        total_weight = out[1]
        dout_x = dout[0]
        dx = nll_loss_grad(x, dout_x, target, weight, total_weight)
        return dx, zeros_like(target), zeros_like(weight)

    return bprop


@bprop_getters.register(P.SparseSoftmaxCrossEntropyWithLogits)
def get_bprop_sparse_softmax_cross_entropy_with_logits(self):
    """Grad definition for `SparseSoftmaxCrossEntropyWithLogits` operation."""
    is_grad = self.is_grad
    grad_op = P.SparseSoftmaxCrossEntropyWithLogits(is_grad=True)

    def bprop(logits, labels, out, dout):
        grad = out[0]
        if not is_grad:
            # if construct use loss
            grad = grad_op(logits, labels)
            grad = F.depend(grad, out)
            grad = grad * dout
        return grad, zeros_like(labels)

    return bprop


@bprop_getters.register(P.ResizeBilinear)
def get_bprop_resize_bilinear(self):
    """Grad definition for `ResizeBilinear` operation."""
    resize_grad = G.ResizeBilinearGrad(self.align_corners, self.half_pixel_centers)

    def bprop(x, out, dout):
        dx = resize_grad(dout, x)
        return (dx,)

    return bprop


@bprop_getters.register(P.OneHot)
def get_bprop_onehot(self):
    """Grad definition for `OneHot` operation."""

    def bprop(indices, depth, on_value, off_value, out, dout):
        return zeros_like(indices), zeros_like(depth), zeros_like(on_value), zeros_like(off_value)

    return bprop


@bprop_getters.register(P.TopK)
def get_bprop_top_kv2(self):
    """Grad definition for `TopK` operation."""
    scatter = P.ScatterNd()
    expand_dims = P.ExpandDims()
    shape_op = P.Shape()
    reshape_op = P.Reshape()
    dtype = P.DType()

    def bprop(input_x, k, out, dout):
        in_shape = shape_op(input_x)
        in_lastdim = in_shape[-1]

        indices = out[1]
        ind_shape = shape_op(indices)
        ind_lastdim = ind_shape[-1]

        ind_2d = reshape_op(indices, (-1, ind_lastdim))
        outerdim = shape_op(ind_2d)[0]

        # [0, outterdim, 2*outerdim, ..., (k-1)*outerdim]
        indices_dtype = dtype(indices)
        range_flatten_index = range_op(0, outerdim * in_lastdim, in_lastdim, indices_dtype)

        # expand_dims to (k, 1), then broadcast
        ind = reshape_op(ind_2d + expand_dims(range_flatten_index, -1), (-1,))
        in_shape_1d = get_1d_shape(in_shape)

        out_grad = reshape_op(
            scatter(
                expand_dims(ind, -1),
                reshape_op(dout[0], (-1,)),
                in_shape_1d),
            in_shape)
        return out_grad, zeros_like(k)

    return bprop


@bprop_getters.register(P.SmoothL1Loss)
def get_bprop_smooth_l1_loss(self):
    """Grad definition for `SmoothL1Loss` operation."""
    grad = G.SmoothL1LossGrad(self.beta)

    def bprop(prediction, target, out, dout):
        dx = grad(prediction, target, dout)
        dy = grad(target, prediction, dout)
        return dx, dy

    return bprop


@bprop_getters.register(P.L2Loss)
def get_bprop_l2_loss(self):
    """Grad definition for `L2Loss` operation."""

    def bprop(x, out, dout):
        dx = x * dout
        return (dx,)

    return bprop


@bprop_getters.register(P.RNNTLoss)
def get_bprop_rnnt_loss(self):
    """Grad definition for `RNNTLoss` operation."""

    def bprop(acts, labels, act_lens, label_lens, out, dout):
        grad = out[1]
        return grad, zeros_like(labels), zeros_like(act_lens), zeros_like(label_lens)

    return bprop


@bprop_getters.register(P.PReLU)
def get_bprop_prelu(self):
    """Grad definition for `PReLU` operation."""
    grad = G.PReLUGrad()

    def bprop(x, w, out, dout):
        dx, dw = grad(dout, x, w)
        return dx, dw

    return bprop


@bprop_getters.register(P.LSTM)
def get_bprop_lstm(self):
    """Grad definition for `LSTM` operation."""
    lstm_grad_data = G.LSTMGradData(
        input_size=self.input_size,
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        has_bias=self.has_bias,
        bidirectional=self.bidirectional,
        dropout=self.dropout
    )

    lstm_grad_weight = G.LSTMGradWeight(
        input_size=self.input_size,
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        has_bias=self.has_bias,
        bidirectional=self.bidirectional,
        dropout=self.dropout
    )
    lstm_grad = G.LSTMGrad(
        input_size=self.input_size,
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        has_bias=self.has_bias,
        bidirectional=self.bidirectional,
        dropout=self.dropout
    )

    def bprop(x, hx, cx, w, out, dout):
        y, _, _, reserve, state = out
        dy, dhy, dcy, _, _ = dout
        dx, dhx, dcx = lstm_grad_data(y, dy, dhy, dcy, w, hx, cx, reserve, state)
        dw = lstm_grad_weight(F.depend(x, dx), hx, y, reserve, state)
        return dx, dhx, dcx, dw

    #
    def bprop_cpu(x, hx, cx, w, out, dout):
        y, hy, cy, reserve, _ = out
        dy, dhy, dcy, _, _ = dout
        dx, dhx, dcx, dw = lstm_grad(x, hx, cx, w, y, hy, cy, dy, dhy, dcy, reserve)
        return dx, dhx, dcx, dw

    if context.get_context('device_target') == "CPU":
        return bprop_cpu

    return bprop


@bprop_getters.register(rl_ops.CudnnGRU)
def get_bprop_gru(self):
    """Grad definition for `GRU` operation."""
    gru_grad_data = G.GruGradData(
        input_size=self.input_size,
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        has_bias=self.has_bias,
        bidirectional=self.bidirectional,
        dropout=self.dropout
    )

    gru_grad_weight = G.GruGradWeight(
        input_size=self.input_size,
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        has_bias=self.has_bias,
        bidirectional=self.bidirectional,
        dropout=self.dropout
    )

    def bprop(x, hx, w, out, dout):
        y, _, reserve, state = out
        dy, dhy, _, _ = dout
        dx, dhx = gru_grad_data(y, dy, dhy, w, hx, reserve, state)
        dw = gru_grad_weight(F.depend(x, dx), hx, y, reserve, state)
        return dx, dhx, dw

    return bprop


@bprop_getters.register(P.DynamicRNN)
def get_bprop_dynamic_rnn(self):
    """Grad definition for `DynamicRNN` operation."""
    dynamic_rnn_grad = G.DynamicRNNGrad(cell_type=self.cell_type,
                                        direction=self.direction,
                                        cell_depth=self.cell_depth,
                                        use_peephole=self.use_peephole,
                                        keep_prob=self.keep_prob,
                                        cell_clip=self.cell_clip,
                                        num_proj=self.num_proj,
                                        time_major=self.time_major,
                                        forget_bias=self.forget_bias)
    expand_dims = P.ExpandDims()

    def bprop(x, w, b, seq_length, init_h, init_c, out, dout):
        dy, dh, dc, _, _, _, _, _, = dout
        dh = dh[-1]
        dc = dc[-1]
        y, h, c, i, j, f, o, tanhct = out
        dw, db, dx, dh_prev, dc_prev = dynamic_rnn_grad(x, w, b, y, init_h[0], init_c[0], h,
                                                        c, dy, dh, dc, i, j, f, o, tanhct)
        dh_prev = expand_dims(dh_prev, 0)
        dc_prev = expand_dims(dc_prev, 0)
        return dx, dw, db, (0), dh_prev, dc_prev

    return bprop


@bprop_getters.register(P.DynamicGRUV2)
def get_bprop_dynamic_gru_v2(self):
    """Grad definition for `DynamicGRUV2` operation."""
    dynamic_gru_v2_grad = G.DynamicGRUV2Grad(self.direction, self.cell_depth, self.keep_prob, self.cell_clip,
                                             self.num_proj, self.time_major, self.gate_order,
                                             self.reset_after)

    def bprop(x, winput, whidden, binput, bhidden, seq, init_h, out, dout):
        y, out_h, update, reset, new, hidden_new = out
        dy, dout_h, _, _, _, _ = dout

        dw_input, dw_hidden, db_input, db_hidden, dx, dh_prev = dynamic_gru_v2_grad(x, winput, whidden, y, init_h,
                                                                                    out_h, dy, dout_h[-1], update,
                                                                                    reset, new, hidden_new, None, None)
        return dx, dw_input, dw_hidden, db_input, db_hidden, (0), dh_prev

    return bprop


@bprop_getters.register(P.SigmoidCrossEntropyWithLogits)
def get_bprop_sigmoid_crossentropy_with_logits(self):
    """Grad definition for `SigmoidCrossEntropyWithLogits` operation."""
    op = G.SigmoidCrossEntropyWithLogitsGrad()

    def bprop(x, y, out, dout):
        dx = op(x, y, dout)
        return (dx, zeros_like(y))

    return bprop


@bprop_getters.register(P.Pad)
def get_bprop_pad(self):
    """Grad definition for `Pad` operation."""
    shape_op = P.Shape()
    paddings = self.paddings

    def bprop(x, out, dout):
        begin = ()
        for item in paddings:
            begin += (item[0],)
        shp = shape_op(x)
        dx = P.Slice()(dout, begin, shp)
        return (dx,)

    return bprop


@bprop_getters.register(P.MirrorPad)
def get_bprop_mirror_pad(self):
    """Grad definition for `MirrorPad` operation."""
    mirror_pad_grad = G.MirrorPadGrad(self.mode)

    def bprop(x, paddings, out, dout):
        dx = mirror_pad_grad(dout, paddings)
        return (dx, zeros_like(paddings))

    return bprop


@bprop_getters.register(P.ROIAlign)
def get_bprop_roi_align(self):
    """Grad definition for `ROIAlign` operation."""
    shape_op = P.Shape()
    pooled_height = self.pooled_height
    pooled_width = self.pooled_width
    spatial_scale = self.spatial_scale
    sample_num = self.sample_num

    def bprop(inputs, rois, out, dout):
        inputs_shape = shape_op(inputs)
        dx = G.ROIAlignGrad(inputs_shape,
                            pooled_height,
                            pooled_width,
                            spatial_scale,
                            sample_num,
                            )(dout, rois)
        return dx, zeros_like(rois)

    return bprop


@bprop_getters.register(P.Conv2DTranspose)
@bprop_getters.register(P.Conv2DBackpropInput)
def get_bprop_conv2d_backprop_input(self):
    """Grad definition for `Conv2DBackpropInput` operation."""
    pad_list = self.get_attr_dict()['pad_list']
    out_channel = self.get_attr_dict()['out_channel']
    filter_grad = G.Conv2DBackpropFilter(
        out_channel, self.kernel_size, self.pad_mode, self.pad, pad_list, mode=self.mode,
        dilation=self.dilation, stride=self.stride, group=self.group, data_format=self.format
    )
    input_grad = P.Conv2D(
        out_channel, self.kernel_size, pad_mode=self.pad_mode.lower(), pad=self.pad,
        dilation=self.dilation, stride=self.stride, group=self.group, data_format=self.format
    )
    get_shape = P.Shape()
    get_dyn_shape = P.TensorShape()

    def bprop(x, w, f_sizes, out, dout):
        w_shape = get_shape(w)
        if -1 in w_shape:
            w_shape = get_dyn_shape(w)
        dx = input_grad(dout, w)
        dw = filter_grad(x, dout, w_shape)
        return dx, dw, zeros_like(f_sizes)

    return bprop


@bprop_getters.register(P.BinaryCrossEntropy)
def get_bprop_binary_cross_entropy(self):
    """Grad definition for `BinaryCrossEntropy` operation."""
    grad = G.BinaryCrossEntropyGrad(self.reduction)

    def bprop(x, y, weight, out, dout):
        dx = grad(x, y, dout, weight)
        return dx, zeros_like(y), zeros_like(weight)

    return bprop


@bprop_getters.register(P.BCEWithLogitsLoss)
def get_bprop_ce_with_logits_loss(self):
    """Grad definition for `BCEWithLogitsLoss` operation."""
    reduction = self.reduction
    mul = P.Mul()
    sigmoid = P.Sigmoid()
    add = P.Add()
    sub = P.Sub()
    size = P.Size()
    neg = P.Neg()
    log = P.Log()

    def bprop(predict, target, weight, pos_weight, out, dout):
        sigmoid_input = sigmoid(predict)
        if pos_weight is not None:
            t = mul(target, pos_weight)
            dx = mul(sub(mul(sub(add(t, 1), target), sigmoid_input), t), dout)
            grad_target = mul(sub(log(sub(1, sigmoid_input)), mul(pos_weight, log(sigmoid_input))), dout)
        else:
            dx = mul((sigmoid_input - target), dout)
            grad_target = mul(predict, neg(dout))
        if weight is not None:
            dx = mul(dx, weight)
            grad_target = mul(grad_target, weight)
        if reduction == 'mean':
            dx = dx / size(dx)
            grad_target = grad_target / size(target)
        return dx, grad_target, zeros_like(weight), zeros_like(pos_weight)

    return bprop


@bprop_getters.register(P.KLDivLoss)
def get_bprop_kl_div_loss(self):
    """Grad definition for `KLDivLoss` operation."""
    grad = G.KLDivLossGrad(self.reduction)

    def bprop(x, y, out, dout):
        dx, dy = grad(x, y, dout)
        return dx, dy

    return bprop


@bprop_getters.register(P.Dropout)
def get_bprop_dropout(self):
    """Grad definition for `Dropout` operation."""
    grad = G.DropoutGrad(self.keep_prob)

    def bprop(x, out, dout):
        _, mask = out
        dy, _ = dout
        dx = grad(dy, mask)
        return (dx,)

    return bprop


@bprop_getters.register(P.Dropout2D)
@bprop_getters.register(P.Dropout3D)
def get_bprop_dropout3d(self):
    """Grad definition for `Dropout2D` and `Dropout3D` operation."""
    dtype = P.DType()
    cast = P.Cast()
    mul = P.Mul()
    keep_prob = self.keep_prob

    def bprop(x, out, dout):
        _, mask = dout
        y = cast(mask, mstype.float32)
        if keep_prob != 0:
            y = y * (1 / keep_prob)
        y = mul(x, y)
        y = cast(y, dtype(x))
        return (y,)

    return bprop


@bprop_getters.register(P.CTCLoss)
def get_bprop_ctc_loss(self):
    """Grad definition for `CTCLoss` operation"""
    expand = P.ExpandDims()

    def bprop(inputs, labels_indices, labels_values, sequence_length, out, dout):
        grad_loss = out[1]
        grad = grad_loss * expand(dout[0], -1)
        return grad, zeros_like(labels_indices), zeros_like(labels_values), zeros_like(sequence_length)

    return bprop


@bprop_getters.register(P.BasicLSTMCell)
def get_bprop_basic_lstm_cell(self):
    """Grad definition for `BasicLSTMCell` operation."""
    basic_lstm_cell_cstate_grad = G.BasicLSTMCellCStateGrad(
        forget_bias=self.forget_bias,
        activation=self.activation
    )

    basic_lstm_cell_weight_grad = G.BasicLSTMCellWeightGrad()

    basic_lstm_cell_input_grad = G.BasicLSTMCellInputGrad(keep_prob=self.keep_prob)

    def bprop(x, h, c, w, b, out, dout):
        _, _, it, jt, ft, ot, tanhct = out
        dct, dht, _, _, _, _, _ = dout
        dgate, dct_1 = basic_lstm_cell_cstate_grad(c, dht, dct, it, jt, ft, ot, tanhct)
        dxt, dht = basic_lstm_cell_input_grad(dgate, w)
        dw, db = basic_lstm_cell_weight_grad(F.depend(x, dxt), h, dgate)
        return dxt, dht, dct_1, dw, db

    return bprop


@bprop_getters.register(P.LRN)
def get_bprop_lrn(self):
    """Grad definition for `LRN` operation."""
    grad = G.LRNGrad(self.depth_radius, self.bias, self.alpha, self.beta)

    def bprop(x, out, dout):
        dx = grad(dout, x, out)
        return (dx,)

    return bprop
