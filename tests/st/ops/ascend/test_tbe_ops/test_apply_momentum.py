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

import luojianet_ms.context as context
import luojianet_ms.nn as nn
from luojianet_ms.common.initializer import initializer
from luojianet_ms.common.parameter import Parameter
from luojianet_ms.ops import operations as P


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.apply_momentum = P.ApplyMomentum(gradient_scale=1024.0)
        self.variable = Parameter(initializer(
            'normal', [2, 3, 3, 4]), name='variable')
        self.accumulation = Parameter(initializer(
            'normal', [2, 3, 3, 4]), name='accumulation')
        self.learning_rate = Parameter(initializer(
            'normal', [1,]), name='learning_rate')
        self.gradient = Parameter(initializer(
            'normal', [2, 3, 3, 4]), name='gradient')
        self.momentum = Parameter(initializer(
            'normal', [1,]), name='momentum')

    def call(self):
        return self.apply_momentum(self.variable, self.accumulation, self.learning_rate, self.gradient, self.momentum)


def test_net():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    apply_momentum = Net()
    output = apply_momentum()
    print(output.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    apply_momentum = Net()
    output = apply_momentum()
    print(output.asnumpy())
