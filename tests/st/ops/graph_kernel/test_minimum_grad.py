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

import numpy as np
import pytest
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops.operations._grad_ops as G


class MinmumGradNet(Cell):
    def __init__(self):
        super(MinmumGradNet, self).__init__()
        self.minimum_grad = G.MinimumGrad()

    def construct(self, x, y, dy):
        return self.minimum_grad(x, y, dy)


def gen_data():
    np.random.seed(0)
    input_x_np = np.random.normal(0, 1, [2, 3]).astype(np.float32)
    input_y_np = np.random.normal(0, 1, [1]).astype(np.float32)
    input_dout_np = np.minimum(input_x_np, input_y_np).astype(np.float32)
    input_x = Tensor(input_x_np)
    input_y = Tensor(input_y_np)
    input_dout = Tensor(input_dout_np)
    return input_x, input_y, input_dout


def get_minimum_grad_output(input_x, input_y, input_dout, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = MinmumGradNet()
    result = net(input_x, input_y, input_dout)
    return result[0].asnumpy(), result[1].asnumpy()


def test_minimum_grad():
    input_x, input_y, input_dout = gen_data()
    result_off = get_minimum_grad_output(input_x, input_y, input_dout, False)
    result_on = get_minimum_grad_output(input_x, input_y, input_dout, True)
    assert np.allclose(result_on[0], result_off[0], rtol=1.e-4, atol=1.e-8, equal_nan=True)
    assert np.allclose(result_on[1], result_off[1], rtol=1.e-4, atol=1.e-8, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_minimum_grad_gpu():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_minimum_grad()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_minimum_grad_ascend():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_minimum_grad()
