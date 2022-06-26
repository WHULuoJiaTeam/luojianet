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
"""
test nn.Triu()
"""
import os
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)

class TriuNet(nn.Cell):
    def __init__(self):
        super(TriuNet, self).__init__()
        self.value = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def construct(self):
        triu = nn.Triu()
        return triu(self.value, 0)

def test_triu():
    """
    Feature: None
    Description: test TriuNet with vm backend
    Expectation: None
    """
    net = TriuNet()
    out = net()
    assert np.sum(out.asnumpy()) == 26

def test_triu_ge():
    """
    Feature: unify ge and vm backend
    Description: test TriuNet with ge backend
    Expectation: None
    """
    os.environ['MS_ENABLE_GE'] = "1"
    os.environ['MS_GE_TRAIN'] = "0"
    net = TriuNet()
    out = net()
    del os.environ['MS_GE_TRAIN']
    del os.environ['MS_ENABLE_GE']
    assert np.sum(out.asnumpy()) == 26

def test_triu_1():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        def construct(self):
            triu = nn.Triu()
            return triu(self.value, 1)

    net = Net()
    out = net()
    assert np.sum(out.asnumpy()) == 11


def test_triu_2():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        def construct(self):
            triu = nn.Triu()
            return triu(self.value, -1)

    net = Net()
    out = net()
    assert np.sum(out.asnumpy()) == 38


def test_triu_parameter():
    class Net(nn.Cell):
        def construct(self, x):
            triu = nn.Triu()
            return triu(x, 0)

    net = Net()
    net(Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


def test_triu_parameter_1():
    class Net(nn.Cell):
        def construct(self, x):
            triu = nn.Triu()
            return triu(x, 1)

    net = Net()
    net(Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


def test_triu_parameter_2():
    class Net(nn.Cell):
        def construct(self, x):
            triu = nn.Triu()
            return triu(x, -1)

    net = Net()
    net(Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
