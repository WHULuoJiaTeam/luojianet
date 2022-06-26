# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Operators for scipy submodule"""
from ..ops import PrimitiveWithInfer, prim_attr_register
from .._checkparam import Validator as validator
from ..common import dtype as mstype


class SolveTriangular(PrimitiveWithInfer):
    """
    Solve the equation `a x = b` for `x`, assuming a is a triangular matrix.

    Args:
        a (Tensor): A triangular matrix of shape :math:`(..., N, N)`.
        b (Tensor): A Tensor of shape :math:`(M,)` or :math:`(..., N, M)`.
            Right-hand side matrix in :math:`a x = b`.
        lower (bool, optional): Use only data contained in the lower triangle of `a`.
            Default is to use upper triangle.
        trans (0, 1, 2, 'N', 'T', 'C', optional):
            Type of system to solve:
            trans:        system:
                0 or 'N'        a x  = b
                1 or 'T'        a^T x = b
                2 or 'C'        a^H x = b
        unit_diagonal (bool, optional): If True, diagonal elements of :math:`a` are assumed to be 1 and
            will not be referenced.
        overwrite_b (bool, optional): Allow overwriting data in :math:`b` (may enhance performance)
        check_finite (bool, optional): Whether to check that the input matrices contain only finite numbers.
            Disabling may give a performance gain, but may result in problems
            (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns:
        Tensor of shape :math:`(..., M,)` or :math:`(..., M, N)`,
        which is the solution to the system :math:`a x = b`.
        Shape of :math:`x` matches :math:`b`.

    Raises:
        LinAlgError: If :math:`a` is singular

    Supported Platforms:
        ``CPU`` ``GPU``

    Examples:
        Solve the lower triangular system :math:`a x = b`, where:

                 [3  0  0  0]       [4]
            a =  [2  1  0  0]   b = [2]
                 [1  0  1  0]       [4]
                 [1  1  1  1]       [2]

        >>> import numpy as onp
        >>> from mindspore.common import Tensor
        >>> import mindspore.numpy as mnp
        >>> from mindspore.scipy.ops import SolveTriangular
        >>> a = Tensor(onp.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], onp.float64))
        >>> b = Tensor(onp.array([4, 2, 4, 2], onp.float64))
        >>> solve_triangular = SolveTriangular(lower=True, unit_diagonal=False, trans='N')
        >>> x = solve_triangular(a, b)
        >>> print(x)
        [ 1.33333333 -0.66666667  2.66666667 -1.33333333]
        >>> print(mnp.dot(a, x))  # Check the result
        [4. 2. 4. 2.]
    """

    @prim_attr_register
    def __init__(self, lower: bool = False, unit_diagonal: bool = False, trans: str = 'N'):
        """Initialize SolveTriangular"""
        super(SolveTriangular, self).__init__("SolveTriangular")
        self.lower = validator.check_value_type(
            "lower", lower, [bool], self.name)
        self.unit_diagonal = validator.check_value_type(
            "unit_diagonal", unit_diagonal, [bool], self.name)
        self.trans = validator.check_value_type(
            "trans", trans, [str], self.name)

        self.init_prim_io_names(inputs=['a', 'b'], outputs=['output'])

    def __infer__(self, a, b):
        a_shape = a['shape']
        b_shape = b['shape']
        # shape match
        b_vector = len(b_shape) == len(a_shape) - 1
        if len(a_shape) < 2:
            raise ValueError(f"For '{self.name}', the dimension of `a` should be at least 2,"
                             f" but got {len(a_shape)} dimensions.")
        b_len = 1 if b_vector else 2
        if len(b_shape) < b_len:
            raise ValueError(f"For '{self.name}', the dimension of `b` should be at least {b_len},"
                             f" but got {len(b_shape)} dimensions.")
        if len(a_shape) != len(b_shape) and len(a_shape) - 1 != len(b_shape):
            raise ValueError(f"For '{self.name}', the dimension of `b` should be 'a.dim' or 'a.dim' - 1, "
                             f"which is {len(a_shape)} or {len(a_shape) - 1}, but got {len(b_shape)} dimensions.")
        if a_shape[-1] != a_shape[-2]:
            raise ValueError(f"For '{self.name}', the last two dimensions of `a` should be the same,"
                             f" but got shape of {a_shape}."
                             f" Please make sure that the shape of `a` be like [..., N, N]")
        if a_shape[-2] != b_shape[-b_len]:
            raise ValueError(f"For '{self.name}', the last two dimensions of `a` and `b` should be matched,"
                             f" but got shape of {a_shape} and {b_shape}."
                             f" Please make sure that the shape of `a` and `b` be like"
                             f" [..., N, N] X [..., N, M] or [..., N, N] X [..., N].")
        if a_shape[:-2] != b_shape[:-b_len]:
            raise ValueError(f"For '{self.name}', the batch dimensions of `a` and `b` should all be the same,"
                             f" but got shape of {a_shape} and {b_shape}."
                             f" Please make sure that the shape of `a` and `b` be like"
                             f" [a, b, c, ..., N, N] X [a, b, c, ..., N, M] or"
                             f" [a, b, c, ..., N, N] X [a, b, c, ..., N].")

        validator.check_scalar_or_tensor_types_same({"a_dtype": a['dtype'], "b_dtype": b['dtype']},
                                                    [mstype.float32, mstype.float64], self.name)
        return {
            'shape': tuple(b_shape),
            'dtype': a['dtype'],
            'value': None
        }

    def infer_dtype(self, a_dtype, b_dtype):
        del b_dtype
        return a_dtype


class Cholesky(PrimitiveWithInfer):
    """
    Inner API Cholesky Compute the Cholesky decomposition of a matrix.
    clean is a special args for mindspore.scipy to indicate whether clean useless triangular matrix data.
    """

    @prim_attr_register
    def __init__(self, clean=True):
        super().__init__("Cholesky")
        self.init_prim_io_names(inputs=['a'], outputs=['l'])
        self.clean = validator.check_value_type("clean", clean, [bool], self.name)
        self.add_prim_attr('clean', self.clean)

    def __infer__(self, a):
        a_shape = a['shape']
        out_shape = a_shape
        output = {
            'shape': tuple(out_shape),
            'dtype': a['dtype'],
            'value': None
        }
        return output


class Eigh(PrimitiveWithInfer):
    """
    Eigh decomposition(Symmetric matrix)
    Ax = lambda * x
    """

    @prim_attr_register
    def __init__(self, compute_eigenvectors=True, lower=True):
        super().__init__(name="Eigh")
        self.init_prim_io_names(inputs=['A'], outputs=['output_w', 'output_v'])
        self.compute_eigenvectors = validator.check_value_type(
            "compute_eigenvectors", compute_eigenvectors, [bool], self.name)
        self.lower = validator.check_value_type("lower", lower, [bool], self.lower)
        self.add_prim_attr('lower', self.lower)
        self.add_prim_attr('compute_eigenvectors', self.compute_eigenvectors)

    def __infer__(self, A):
        validator.check_scalar_or_tensor_types_same({"A_dtype": A['dtype']},
                                                    [mstype.float32, mstype.float64, mstype.complex64,
                                                     mstype.complex128], self.name, True)
        output = None
        if self.compute_eigenvectors:
            output = {
                'shape': ((A['shape'][0],), (A['shape'][0], A['shape'][0])),
                'dtype': (A['dtype'], A['dtype']),
                'value': None
            }
        else:
            output = {
                'shape': (A['shape'][0],),
                'dtype': A['dtype'],
                'value': None
            }
        return output


class Eig(PrimitiveWithInfer):
    """
    Eig decomposition,(generic matrix)
    a * v = w * v
    """

    @prim_attr_register
    def __init__(self, compute_v=True):
        super().__init__(name="Eig")
        self.init_prim_io_names(inputs=['a'], outputs=['w', 'v'])
        self.compute_v = validator.check_value_type("compute_v", compute_v, [bool], self.name)
        self.add_prim_attr('compute_v', self.compute_v)
        self.io_table = {
            mstype.tensor_type(mstype.float32): mstype.complex64,
            mstype.tensor_type(mstype.complex64): mstype.complex64,
            mstype.tensor_type(mstype.float64): mstype.complex128,
            mstype.tensor_type(mstype.complex128): mstype.complex128
        }

    def __infer__(self, a):
        a_dtype = a["dtype"]
        a_shape = tuple(a["shape"])
        validator.check_tensor_dtype_valid("a", a_dtype,
                                           [mstype.float32, mstype.float64, mstype.complex64, mstype.complex128],
                                           self.name)

        output = None
        if self.compute_v:
            output = {
                'shape': (a_shape[:-1], a_shape),
                'dtype': (self.io_table.get(a_dtype), self.io_table.get(a_dtype)),
                'value': None
            }
        else:
            output = {
                'shape': a_shape[:-1],
                'dtype': self.io_table.get(a_dtype),
                'value': None
            }
        return output


class LU(PrimitiveWithInfer):
    """
    LU decomposition with partial pivoting
    A = P.L.U
    """

    @prim_attr_register
    def __init__(self):
        super().__init__(name="LU")
        self.init_prim_io_names(inputs=['x'], outputs=['lu', 'pivots', 'permutation'])

    def __infer__(self, x):
        x_shape = list(x['shape'])
        x_dtype = x['dtype']
        k_shape = min(x_shape[-1], x_shape[-2])
        permutation_shape = x_shape[:-2] + [k_shape, k_shape]
        pivots_shape = x_shape[:-2] + [k_shape]
        output = {
            'shape': (x_shape, pivots_shape, permutation_shape),
            'dtype': (x_dtype, mstype.int32, mstype.int32),
            'value': None
        }
        return output


class MatrixSetDiag(PrimitiveWithInfer):
    """
    Inner API to set a [..., M, N] matrix's diagonals by range[k[0], k[1]].
    """

    @prim_attr_register
    def __init__(self, alignment: str):
        super().__init__(name="MatrixSetDiag")
        self.init_prim_io_names(inputs=['input_x', 'diagonal', 'k'], outputs=['output'])
        self.alignment = validator.check_value_type("alignment", alignment, [str], self.name)

    def __infer__(self, input_x, diagonal, k):
        in_shape = list(input_x['shape'])
        in_dtype = input_x['dtype']
        output = {
            'shape': tuple(in_shape),
            'dtype': in_dtype,
            'value': None
        }
        return output


class MatrixBandPart(PrimitiveWithInfer):
    """
    MatrixBandPart
    """

    @prim_attr_register
    def __init__(self):
        super().__init__(name="MatrixBandPart")
        self.init_prim_io_names(inputs=['A', 'lower_numer', 'upper_number'], outputs=['output'])

    def __infer__(self, a, lower, upper):
        shape = {
            'shape': (a['shape']),
            'dtype': (a['dtype']),
            'value': None
        }
        return shape


class MatrixDiagPartV3(PrimitiveWithInfer):
    """
    Returns:
        batched diagonal part of a batched tensor, the part between, k[0] to k[1], the shape is dynamic
    Raises:
        k[1] should not less then k[0], or padding value dtype is not same with input tensor A.dtype
    """

    @prim_attr_register
    def __init__(self, align="RIGHT_LEFT"):
        super().__init__(name="MatrixDiagPartV3")
        self.add_prim_attr('alignment', align)
        self.init_prim_io_names(inputs=['A', 'k', 'padding_value'], outputs=['output'])


# pylint: disable=C0413,W0611
from .ops_grad import get_bprop_cholesky, get_bprpo_eigh, get_bprpo_trsm
