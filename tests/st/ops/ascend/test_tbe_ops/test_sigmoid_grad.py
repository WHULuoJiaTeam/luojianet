# Copyright 2019 Huawei Technologies Co., Ltd
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

context.set_context(device_target="Ascend")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.sigmoid = P.Sigmoid()

    @ms_function
    def forward(self, x):
        return self.sigmoid(x)


class Grad(nn.Module):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    @ms_function
    def forward(self, x, y):
        return self.grad(self.network)(x, y)


def test_net():
    x = np.random.random(size=(2, 3, 4, 5, 6)).astype(np.float32)
    y = np.random.random(size=(2, 3, 4, 5, 6)).astype(np.float32)
    net = Grad(Net())
    output = net(Tensor(x), Tensor(y))
    print("=================output====================")
    print(output.asnumpy())
