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
from luojianet_ms.common.api import ms_function
from luojianet_ms.ops import operations as P
from luojianet_ms.ops.composite import GradOperation

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Grad(nn.Module):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    @ms_function
    def forward(self, input_, output_grad):
        return self.grad(self.network)(input_, output_grad)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.HSigmoid = P.HSigmoid()

    def forward(self, x):
        return self.HSigmoid(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net():
    x = np.array([-1, -2, 0, 2, 1]).astype(np.float32)
    hswish = Net()
    y = hswish(Tensor(x))
    expect = np.array([0.33333334, 0.16666667, 0.5, 0.8333333, 0.6666667]).astype(np.float32)
    assert np.all(y.asnumpy() == expect)
    sens = np.random.randn(5).astype(np.float32)
    backword_net = Grad(Net())
    output = backword_net(Tensor(x), Tensor(sens))
    print(len(output))
    print(output[0].asnumpy())
