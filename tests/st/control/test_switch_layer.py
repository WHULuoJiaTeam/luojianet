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
from mindspore import Tensor, nn
from mindspore.common import dtype as mstype
from mindspore.ops.composite import GradOperation


class Grad(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.grad = GradOperation(get_all=False)
        self.net = net

    def construct(self, x, y):
        grad_net = self.grad(self.net)
        grad = grad_net(x, y)
        return grad


class CaseNet(nn.Cell):
    def __init__(self):
        super(CaseNet, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3)
        self.relu = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.softmax = nn.Softmax()
        self.layers1 = (self.relu, self.softmax)
        self.layers2 = (self.conv, self.relu1)

    def construct(self, x, index1, index2):
        x = self.layers1[index1](x)
        x = self.layers2[index2](x)
        return x


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_switch_layer():
    context.set_context(mode=context.GRAPH_MODE)
    net = CaseNet()
    data = Tensor(np.ones((1, 1, 224, 224)), mstype.float32)
    idx = Tensor(0, mstype.int32)
    idx2 = Tensor(-1, mstype.int32)
    value = net(data, idx, idx2)
    relu = nn.ReLU()
    true_value = relu(data)
    ret = np.allclose(value.asnumpy(), true_value.asnumpy())
    assert ret


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cell_in_list():
    """
    Feature: Switch layer in while.
    Description: test recursive switch layer.
    Expectation: success if grad and output are correct.
    """

    class TestCell(nn.Cell):
        def __init__(self, i):
            super().__init__()
            self.i = i

        def construct(self, x):
            return self.i * x

    class CellInList(nn.Cell):
        def __init__(self):
            super().__init__()
            self.cell_list = nn.CellList()
            self.cell_list.append(TestCell(4))
            self.cell_list.append(TestCell(5))
            self.cell_list.append(TestCell(6))

        def construct(self, t, x):
            out = t
            while x < 3:
                add = self.cell_list[x](t)
                out = out + add
                x += 1
            return out

    net = CellInList()
    t = Tensor(10, mstype.int32)
    x = Tensor(0, mstype.int32)
    out = net(t, x)
    grad_net = Grad(net)
    grad_out = grad_net(t, x)

    assert out == Tensor(160, mstype.int32)
    assert grad_out == Tensor(16, mstype.int32)
