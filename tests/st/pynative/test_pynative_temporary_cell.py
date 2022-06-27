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
import luojianet_ms.nn as nn
from luojianet_ms import Tensor
import luojianet_ms.ops as P
from luojianet_ms.nn.optim import Momentum
from luojianet_ms.common import ParameterTuple


class GradofParams(nn.Module):
    def __init__(self, net, sens=False):
        super().__init__()
        self.grad = P.GradOperation(get_all=False, get_by_list=True, sens_param=sens)
        self.net = net
        self.params = ParameterTuple(self.net.trainable_params())

    def forward(self, *x):
        out = self.grad(self.net, self.params)(*x)
        return out

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pynative_temporary_cell_variables():
    context.set_context(mode=context.PYNATIVE_MODE)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.conv = nn.Conv2d(1, 1, 3, weight_init='ones', pad_mode='pad')
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.add(x, x)
            return x

    class TempCellNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.conv = nn.Conv2d(1, 1, 3, weight_init='ones', pad_mode='pad')

        def forward(self, x):
            x = self.conv(x)
            x = nn.ReLU()(x)
            x = self.add(x, x)
            return x

    input_data = Tensor(np.random.randn(1, 1, 224, 224).astype(np.float32))
    # The first net run
    net = Net()
    backnet = GradofParams(net)
    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.1, 0.9)
    grad_first = backnet(input_data)
    optimizer(grad_first)
    grad_second = backnet(input_data)
    # The second net run
    compare_net = TempCellNet()
    compare_backnet = GradofParams(compare_net)
    compare_optimizer = Momentum(filter(lambda x: x.requires_grad, compare_net.get_parameters()), 0.1, 0.9)
    compare_grad_first = compare_backnet(input_data)
    compare_optimizer(compare_grad_first)
    compare_grad_second = compare_backnet(input_data)
    # compare result
    assert np.allclose(grad_first[0].asnumpy(), compare_grad_first[0].asnumpy(), 0.01, 0.01)
    assert np.allclose(grad_second[0].asnumpy(), compare_grad_second[0].asnumpy(), 0.01, 0.01)
