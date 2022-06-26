# Copyright 2020 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.sparse_apply_ftrl = P.SparseApplyFtrl(lr=0.001, l1=0.0, l2=0.0, lr_power=-0.5, use_locking=False)
        self.var = Parameter(Tensor(np.ones([3, 3, 3]).astype(np.float32)), name="var")
        self.accum = Parameter(Tensor(np.ones([3, 3, 3]).astype(np.float32)), name="accum")
        self.linear = Parameter(Tensor(np.ones([3, 3, 3]).astype(np.float32)), name="linear")

    def construct(self, grad, indices):
        out = self.sparse_apply_ftrl(self.var, self.accum, self.linear, grad, indices)
        return out


class Net_half(nn.Cell):
    def __init__(self):
        super(Net_half, self).__init__()
        self.sparse_apply_ftrl = P.SparseApplyFtrl(lr=0.001, l1=0.0, l2=0.0, lr_power=-0.5, use_locking=False)
        self.var = Parameter(Tensor(np.ones([3, 3, 3]).astype(np.float16)), name="var")
        self.accum = Parameter(Tensor(np.ones([3, 3, 3]).astype(np.float16)), name="accum")
        self.linear = Parameter(Tensor(np.ones([3, 3, 3]).astype(np.float16)), name="linear")

    def construct(self, grad, indices):
        out = self.sparse_apply_ftrl(self.var, self.accum, self.linear, grad, indices)
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ftrl():
    gradient = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    indices = Tensor([0, 1, 2], mstype.int32)
    expect_var = np.array([[[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]],
                           [[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]],
                           [[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]]]).astype(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    sparse_apply_ftrl = Net()
    sparse_apply_ftrl(gradient, indices)
    assert np.all(sparse_apply_ftrl.var.data.asnumpy() == expect_var)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    sparse_apply_ftrl = Net()
    sparse_apply_ftrl(gradient, indices)
    assert np.all(sparse_apply_ftrl.var.data.asnumpy() == expect_var)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ftrl_sparse_int64_ind():
    gradient = Tensor(np.ones([2, 3, 3]).astype(np.float32))
    indices = Tensor([0, 2], mstype.int64)
    expect_var = np.array([[[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]],
                           [[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]],
                           [[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]]]).astype(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    sparse_apply_ftrl = Net()
    sparse_apply_ftrl(gradient, indices)
    assert np.all(sparse_apply_ftrl.var.data.asnumpy() == expect_var)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    sparse_apply_ftrl = Net()
    sparse_apply_ftrl(gradient, indices)
    assert np.all(sparse_apply_ftrl.var.data.asnumpy() == expect_var)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ftrl_half():
    gradient = Tensor(np.ones([3, 3, 3]).astype(np.float16))
    indices = Tensor([0, 1, 2], mstype.int32)
    expect_var = np.array([[[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]],
                           [[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]],
                           [[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]]]).astype(np.float16)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    sparse_apply_ftrl = Net_half()
    sparse_apply_ftrl(gradient, indices)
    assert np.all(sparse_apply_ftrl.var.data.asnumpy() == expect_var)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    sparse_apply_ftrl = Net_half()
    sparse_apply_ftrl(gradient, indices)
    assert np.all(sparse_apply_ftrl.var.data.asnumpy() == expect_var)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ftrl_sparse_half_int64_ind():
    gradient = Tensor(np.ones([2, 3, 3]).astype(np.float16))
    indices = Tensor([0, 2], mstype.int64)
    expect_var = np.array([[[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]],
                           [[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]],
                           [[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]]]).astype(np.float16)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    sparse_apply_ftrl = Net_half()
    sparse_apply_ftrl(gradient, indices)
    assert np.all(sparse_apply_ftrl.var.data.asnumpy() == expect_var)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    sparse_apply_ftrl = Net_half()
    sparse_apply_ftrl(gradient, indices)
    assert np.all(sparse_apply_ftrl.var.data.asnumpy() == expect_var)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ftrl_half_return_output():
    gradient = Tensor(np.ones([3, 3, 3]).astype(np.float16))
    indices = Tensor([0, 1, 2], mstype.int32)
    expect_var = np.array([[[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]],
                           [[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]],
                           [[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]]]).astype(np.float16)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    sparse_apply_ftrl = Net_half()
    output = sparse_apply_ftrl(gradient, indices)
    assert np.all(output[0].asnumpy() == expect_var)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    sparse_apply_ftrl = Net_half()
    sparse_apply_ftrl(gradient, indices)
    assert np.all(output[0].asnumpy() == expect_var)
