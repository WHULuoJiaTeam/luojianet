# Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
# Copyright 2021, 2022 Huawei Technologies Co., Ltd
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

    Inputs:
        - **sparse_tensor** (CSRTensor) - A CSRTensor.
        - **axis** (int) - The dimensions to reduce.

    Outputs:
        Tensor, the dtype is the same as `sparse_tensor.values`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import luojianet_ms
        >>> import luojianet_ms.nn as nn
        >>> from luojianet_ms import Tensor, CSRTensor, ops
        >>> from luojianet_ms import dtype as mstype
        >>> class Net(nn.Module):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.op = ops.CSRReduceSum()
        ...
        ...     def call(self, indptr, indices, values, dense_shape, axis):
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

    Inputs:
        - **sparse_tensor** (CSRTensor) - A CSRTensor.
        - **dense_tensor** (Tensor) - A dense Tensor.

    Outputs:
        Tensor.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import luojianet_ms
        >>> import luojianet_ms.nn as nn
        >>> from luojianet_ms import Tensor, CSRTensor, ops
        >>> from luojianet_ms import dtype as mstype
        >>> class Net(nn.Module):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.op = ops.CSRMV()
        ...
        ...     def call(self, indptr, indices, values, dense_shape, dense):
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
    Elemwise multiplication on a CSRTensor and a dense tensor.

    Note:
        The op outputs a 1-D dense tensor whose shape and values are the same as input `CSRTensor.values`.
        If expect a CSRTensor output, please use `*` directly, e.g. `x * y`, `x` or `y` can be CSRTensor.

    Inputs:
        - **sparse_tensor** (CSRTensor) - A CSRTensor.
        - **dense_tensor** (Tensor) - A Tensor.

    Outputs:
        Tensor, the dtype and shape is the same as `sparse_tensor.values`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import luojianet_ms
        >>> import luojianet_ms.nn as nn
        >>> from luojianet_ms import Tensor, CSRTensor, ops
        >>> from luojianet_ms import dtype as mstype
        >>> class Net(nn.Module):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.op = ops.CSRMul()
        ...
        ...     def call(self, indptr, indices, values, dense_shape, dense):
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
