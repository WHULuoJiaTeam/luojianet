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

import numpy as np
import pytest
import luojianet_ms.context as context
from luojianet_ms import Tensor
from luojianet_ms.nn import Module
import luojianet_ms.ops.operations._grad_ops as G


class TanhGradNet(Module):
    def __init__(self):
        super(TanhGradNet, self).__init__()
        self.tanh_grad = G.TanhGrad()

    def call(self, y, dy):
        return self.tanh_grad(y, dy)


def test_tanh_grad():
    np.random.seed(0)
    input_y = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    input_dy = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    net = TanhGradNet()
    result = net(Tensor(input_y), Tensor(input_dy))
    expect = input_dy * (1.0 - input_y * input_y)
    res = np.allclose(expect, result.asnumpy(), rtol=1.e-4, atol=1.e-7, equal_nan=True)
    assert res


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tanh_grad_gpu():
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target="GPU")
    test_tanh_grad()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tanh_grad_ascend():
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target="Ascend")
    test_tanh_grad()
