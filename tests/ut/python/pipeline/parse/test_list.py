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
""" test_list """
import numpy as np

import luojianet_ms.nn as nn
from luojianet_ms import Tensor
from luojianet_ms.common.api import _cell_graph_executor
from luojianet_ms.nn import Module


class Net1(Module):
    def __init__(self, list1):
        super().__init__()
        self.list = list1
        self.fla = nn.Flatten()

    def call(self, x):
        for _ in self.list:
            x = self.fla(x)
        return x


def test_list1():
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = Net1([1])
    _cell_graph_executor.compile(net, input_me)


def test_list2():
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = Net1([1, 2])
    _cell_graph_executor.compile(net, input_me)
