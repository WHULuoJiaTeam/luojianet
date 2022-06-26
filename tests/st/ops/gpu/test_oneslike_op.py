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
from mindspore.ops import operations as P

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


class NetOnesLike(nn.Cell):
    def __init__(self):
        super(NetOnesLike, self).__init__()
        self.ones_like = P.OnesLike()

    def construct(self, x):
        return self.ones_like(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_OnesLike():
    x0_np = np.random.uniform(-2, 2, (2, 3, 4, 4)).astype(np.float32)
    x1_np = np.random.uniform(-2, 2, 1).astype(np.float16)
    x2_np = np.zeros([3, 3, 3], dtype=np.int32)

    x0 = Tensor(x0_np)
    x1 = Tensor(x1_np)
    x2 = Tensor(x2_np)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    ones_like = NetOnesLike()
    output0 = ones_like(x0)
    expect0 = np.ones_like(x0_np)
    diff0 = output0.asnumpy() - expect0
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    assert np.all(diff0 < error0)
    assert output0.shape == expect0.shape

    output1 = ones_like(x1)
    expect1 = np.ones_like(x1_np)
    diff1 = output1.asnumpy() - expect1
    error1 = np.ones(shape=expect1.shape) * 1.0e-5
    assert np.all(diff1 < error1)
    assert output1.shape == expect1.shape

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    ones_like = NetOnesLike()
    output0 = ones_like(x0)
    expect0 = np.ones_like(x0_np)
    diff0 = output0.asnumpy() - expect0
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    assert np.all(diff0 < error0)
    assert output0.shape == expect0.shape

    output1 = ones_like(x1)
    expect1 = np.ones_like(x1_np)
    diff1 = output1.asnumpy() - expect1
    error1 = np.ones(shape=expect1.shape) * 1.0e-5
    assert np.all(diff1 < error1)
    assert output1.shape == expect1.shape

    output2 = ones_like(x2)
    expect2 = np.ones_like(x2_np)
    diff2 = output2.asnumpy() - expect2
    error2 = np.ones(shape=expect2.shape) * 1.0e-5
    assert np.all(diff2 < error2)
    assert output2.shape == expect2.shape
