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

import numpy as np

import luojianet_ms as ms
import luojianet_ms.nn as nn
from luojianet_ms import Tensor
from luojianet_ms import context
from luojianet_ms.common.api import _cell_graph_executor
from luojianet_ms.ops import composite as C
from luojianet_ms.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Module):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def forward(self, x, y):
        predict = self.network(x, y)
        return self.loss(predict)


class GradWrap(nn.Module):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def forward(self, x, y):
        return grad_all(self.network)(x, y)


class Net(nn.Module):
    def __init__(self, strategy):
        super().__init__()
        self.reshape = P.Reshape()
        self.mul = P.Mul().shard(strategy)
        self.relu = P.ReLU()

    def forward(self, x, y):
        out = self.reshape(x, (10000, 36, 1))
        out = self.mul(out, y)
        out = self.relu(out)
        return out


def compile_net(net, x, y):
    net.set_auto_parallel()
    net.set_train()
    _cell_graph_executor.compile(net, x, y)


def test_reshape_parameter_data_parallel():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy = ((8, 1, 1), (8, 1, 1))
    net = GradWrap(NetWithLoss(Net(strategy)))
    x = Tensor(np.ones([10000, 36]), dtype=ms.float32)
    y = Tensor(np.ones([10000, 36, 1]), dtype=ms.float32)
    compile_net(net, x, y)


def test_reshape_parameter_model_parallel():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy = ((4, 2, 1), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(strategy)))
    x = Tensor(np.ones([10000, 36]), dtype=ms.float32)
    y = Tensor(np.ones([10000, 36, 1]), dtype=ms.float32)
    compile_net(net, x, y)
