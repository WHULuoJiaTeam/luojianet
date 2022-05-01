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
""" test expand_as"""
import luojianet_ms as ms
import luojianet_ms.nn as nn
import luojianet_ms.common.initializer as init
from luojianet_ms import Tensor
from luojianet_ms import context

context.set_context(mode=context.GRAPH_MODE)


def test_expand_as():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.t1 = Tensor([1, 2, 3])
            self.t2 = Tensor([[1, 1, 1], [1, 1, 1]])

        def call(self):
            return self.t1.expand_as(self.t2)

    net = Net()
    net()


def test_initializer_expand_as():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.t1 = init.initializer('one', [1, 3], ms.float32)
            self.t2 = init.initializer('one', [2, 3], ms.float32)

        def call(self):
            return self.t1.expand_as(self.t2)

    net = Net()
    net()


def test_expand_as_parameter():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.t1 = Tensor([1, 2, 3])

        def call(self, x):
            return self.t1.expand_as(x)

    net = Net()
    net(Tensor([[1, 1, 1], [1, 1, 1]]))


def test_expand_tensor_as_parameter_1():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.t2 = Tensor([[1, 1, 1], [1, 1, 1]])

        def call(self, x):
            return x.expand_as(self.t2)

    net = Net()
    net(Tensor([1, 2, 3]))
