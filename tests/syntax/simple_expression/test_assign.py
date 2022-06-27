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
from luojianet_ms.common.tensor import Tensor

context.set_context(mode=context.GRAPH_MODE)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.m = 1

    def forward(self, x, y):
        x += 1
        #x += self.x
        print(x)
        #x = y
        x = "aaa"
        #x = 5.0
        return x


def test_assign():
    net = Net()
    y = Tensor((1), luojianet_ms.int32)
    x = 1
    ret = net(x, y)
    print(ret)
    print(x)
