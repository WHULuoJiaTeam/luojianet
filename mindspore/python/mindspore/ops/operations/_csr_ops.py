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
"""csr_ops"""
from ..primitive import prim_attr_register, PrimitiveWithInfer


class CSRReduceSum(PrimitiveWithInfer):
    """
    Reduces a dimension of a CSRTensor by summing all elements in the dimension.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Inputs:
        - **sparse_tensor** (CSRTensor) - A CSRTensor.
        - **axis** (int) - The dimensions to reduce.

    Outputs:
        Tensor, the dtype is the same as `sparse_tensor.values`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor, CSRTensor
        >>> from mindspore.ops.operations import _csr_ops
        >>> from mindspore import dtype as mstype
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.op = _csr_ops.CSRReduceSum()
        ...
        ...     def construct(self, indptr, indices, values, dense_shape, axis):
        ...         csr_tensor = CSRTensor(indptr, indices, values, dense_shape)
        ...         return self.op(csr_tensor, axis)
        >>> indptr = Tensor([0, 1, 2])
        >>> indices = Tensor([0, 1])
        >>> values = Tensor([2, 1], dtype=mstype.float32)
        >>> dense_shape = (2, 4)
        >>> out = Net()(indptr, indices, values, dense_shape, 1)
        >>> print(out)
        [[2.]
         [1.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize CSRReduceSum"""
        self.init_prim_io_names(inputs=['indptr', 'indices', 'values', 'dense_shape', 'axis'],
                                outputs=['output'])


class CSRMV(PrimitiveWithInfer):
    """
    Sparse matrix-vector multiplication.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Inputs:
        - **sparse_tensor** (CSRTensor) - A CSRTensor.
        - **dense_tensor** (Tensor) - A dense Tensor.

    Outputs:
        Tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor, CSRTensor
        >>> from mindspore.ops.operations import _csr_ops
        >>> from mindspore import dtype as mstype
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.op = _csr_ops.CSRMV()
        ...
        ...     def construct(self, indptr, indices, values, dense_shape, dense):
        ...         csr_tensor = CSRTensor(indptr, indices, values, dense_shape)
        ...         return self.op(csr_tensor, dense)
        >>> indptr = Tensor([0, 1, 2])
        >>> indices = Tensor([0, 1])
        >>> values = Tensor([2, 1], dtype=mstype.float32)
        >>> dense_shape = (2, 4)
        >>> dense = Tensor([[1], [1], [1], [1]], dtype=mstype.float32)
        >>> out = Net()(indptr, indices, values, dense_shape, dense)
        >>> print(out)
        [[2.]
         [1.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize CSRMV"""
        self.init_prim_io_names(inputs=['indptr', 'indices', 'values', 'dense_shape', 'dense_tensor'],
                                outputs=['output'])


class CSRMul(PrimitiveWithInfer):
    """
    Elemwise multiplication of a CSRTensor and a dense tensor.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Note:
        The op outputs a 1-D dense tensor whose shape and values are the same as input `CSRTensor.values`.
        If expect a CSRTensor output, please use `*` directly, e.g. `x * y`, `x` or `y` can be CSRTensor.

    Inputs:
        - **sparse_tensor** (CSRTensor) - A CSRTensor.
        - **dense_tensor** (Tensor) - A Tensor.

    Outputs:
        Tensor, the dtype and shape is the same as `sparse_tensor.values`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor, CSRTensor
        >>> from mindspore.ops.operations import _csr_ops
        >>> from mindspore import dtype as mstype
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.op = _csr_ops.CSRMul()
        ...
        ...     def construct(self, indptr, indices, values, dense_shape, dense):
        ...         csr_tensor = CSRTensor(indptr, indices, values, dense_shape)
        ...         return self.op(csr_tensor, dense)
        >>> indptr = Tensor([0, 1, 2])
        >>> indices = Tensor([0, 1])
        >>> values = Tensor([2, 1], dtype=mstype.float32)
        >>> dense_shape = (2, 4)
        >>> dense = Tensor([[1., 1, 1, 1], [1, 1, 1, 1]], dtype=mstype.float32)
        >>> out = Net()(indptr, indices, values, dense_shape, dense)
        >>> print(out)
        [2. 1.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize CSRMul"""
        self.init_prim_io_names(inputs=['indptr', 'indices', 'values', 'dense_shape', 'dense_tensor'],
                                outputs=['output'])


class CSRGather(PrimitiveWithInfer):
    """
    Returns the values of a CSRTensor indexed from a dense tensor using indptr and indices.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Inputs:
        - **indptr** (Tensor) - A Tensor.
        - **indices** (Tensor) - A Tensor.
        - **dense** (Tensor) - A Tensor.
        - **sparse_shape** (tuple) - A tuple of integers.

    Outputs:
        Tensor, the dtype is the same as `dense`, the first dimension is the same shape as `indices` and the remaining
    dimensions are the same as ``dense[2:]``.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> from mindspore.ops.operations import _csr_ops
        >>> from mindspore import dtype as mstype
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.op = _csr_ops.CSRGather()
        ...
        ...     def construct(self, indptr, indices, dense, sparse_shape):
        ...         return self.op(indptr, indices, dense, sparse_shape)
        >>> indptr = Tensor([0, 1, 2])
        >>> indices = Tensor([0, 1])
        >>> sparse_shape = (2, 4)
        >>> dense = Tensor([[1., 1, 1, 1], [1, 1, 1, 1]], dtype=mstype.float32)
        >>> out = Net()(indptr, indices, dense, sparse_shape)
        >>> print(out)
        [1. 1.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize CSRGather"""
        self.init_prim_io_names(inputs=['indptr', 'indices', 'dense', 'dense_shape'],
                                outputs=['output'])


class CSR2COO(PrimitiveWithInfer):
    """
    Converts the indptr of a CSRTensor to the row indices of a COOTensor.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Inputs:
        - **indptr** (Tensor) - A Tensor.
        - **nnz** (int) - Denotes the number of non-zero elements in the sparse tensor.

    Outputs:
        Tensor, the dtype is the same as `indptr` and has shape (`nnz`,).

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> from mindspore.ops.operations import _csr_ops
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.op = _csr_ops.CSR2COO()
        ...
        ...     def construct(self, indptr, nnz):
        ...         return self.op(indptr, nnz)
        >>> indptr = Tensor([0, 1, 2])
        >>> out = Net()(indptr, 2)
        >>> print(out)
        [1 1]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize CSR2COO"""
        self.init_prim_io_names(inputs=['indptr', 'nnz'], outputs=['output'])


class COO2CSR(PrimitiveWithInfer):
    """
    Converts the row indices of a COOTensor to the indptr of a CSRTensor.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Inputs:
        - **row_indices** (Tensor) - A Tensor.
        - **height** (int) - the height of the first dimension of the sparse tensor.

    Outputs:
        Tensor, the dtype is the same as `row_indices` and has shape ('height' + 1,).

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> from mindspore.ops.operations import _csr_ops
        >>> from mindspore import dtype as mstype
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.op = _csr_ops.COO2CSR()
        ...
        ...     def construct(self, row_indices, height):
        ...         return self.op(row_indices, height)
        >>> row_indices = Tensor([0, 1])
        >>> out = Net()(row_indices, 2)
        >>> print(out)
        [0 1 2]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize COO2CSR"""
        self.init_prim_io_names(inputs=['row_indices', 'height'], outputs=['output'])


class CSRDiv(PrimitiveWithInfer):
    """
    Elemwise division on a CSRTensor and a dense tensor.

    Note:
        The op outputs a 1-D dense tensor whose shape and values are the same as input `CSRTensor.values`.
        If expect a CSRTensor output, please use `/` directly, e.g. `x / y`, can be CSRTensor.

    Inputs:
        - **sparse_tensor** (CSRTensor) - A CSRTensor.
        - **dense_tensor** (Tensor) - A Tensor.

    Outputs:
        Tensor, the dtype and shape is the same as `sparse_tensor.values`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor, CSRTensor
        >>> from mindspore.ops.operations import _csr_ops
        >>> from mindspore import dtype as mstype
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.op = _csr_ops.CSRDiv()
        ...
        ...     def construct(self, indptr, indices, values, dense_shape, dense):
        ...         csr_tensor = CSRTensor(indptr, indices, values, dense_shape)
        ...         return self.op(csr_tensor, dense)
        >>> indptr = Tensor([0, 1, 2])
        >>> indices = Tensor([0, 1])
        >>> values = Tensor([2, 1], dtype=mstype.float32)
        >>> dense_shape = (2, 4)
        >>> dense = Tensor([[1., 1, 1, 1], [1, 1, 1, 1]], dtype=mstype.float32)
        >>> out = Net()(indptr, indices, values, dense_shape, dense)
        >>> print(out)
        [2. 1.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize CSRDiv"""
        self.init_prim_io_names(inputs=['indptr', 'indices', 'values', 'dense_shape', 'dense_tensor'],
                                outputs=['output'])
