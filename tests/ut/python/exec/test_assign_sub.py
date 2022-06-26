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
"""
test assign sub
"""
import numpy as np

import luojianet_ms.context as context
import luojianet_ms.nn as nn
import luojianet_ms.ops.operations as P
from luojianet_ms import Tensor
from luojianet_ms.common.initializer import initializer
from luojianet_ms.common.parameter import Parameter

context.set_context(mode=context.GRAPH_MODE)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.b = Parameter(initializer('ones', [5]), name='b')
        self.sub = P.AssignSub()

    def construct(self, value):
        return self.sub(self.b, value)


def test_net():
    net = Net()
    input_data = Tensor(np.ones([5]).astype(np.float32))
    output = net(input_data)
    print(output.asnumpy().shape)
    print(output.asnumpy())
