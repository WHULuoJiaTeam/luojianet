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

import luojianet_ms.context as context
import luojianet_ms.nn as nn
from luojianet_ms import Tensor
from luojianet_ms.common.api import ms_function
from luojianet_ms.ops import operations as P
from luojianet_ms.ops.composite import GradOperation

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Grad(nn.Module):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True)
        self.network = network

    @ms_function
    def forward(self, input_):
        return self.grad(self.network)(input_)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu_v2 = P.ReLUV2()

    def forward(self, x):
        return self.relu_v2(x)


def test_net():
    x = Tensor(np.ones((2, 3, 3, 4)).astype(np.float32))
    relu_net = Net()
    relu_output = relu_net(x)
    net = Grad(Net())
    output_grad = net(x)
    print(relu_output[0].asnumpy())
    print(relu_output[1].asnumpy())
    print(len(output_grad))
    print(output_grad[0].asnumpy())
