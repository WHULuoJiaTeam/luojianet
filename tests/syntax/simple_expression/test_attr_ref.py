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
""" test syntax for logic expression """

import luojianet_ms.nn as nn
import luojianet_ms
from luojianet_ms import context
from luojianet_ms.common.parameter import Parameter
from luojianet_ms.common.tensor import Tensor

context.set_context(mode=context.GRAPH_MODE)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter(Tensor(3, luojianet_ms.float32), name="w")
        self.m = 2

    def forward(self, x, y):
        self.weight = x
        self.m = 3
        #self.l = 1
        #y.weight = x
        print(self.weight)
        return x

def test_attr_ref():
    x = Tensor(4, luojianet_ms.float32)
    net_y = Net()
    net = Net()
    ret = net(x, net_y)
    print(ret)
