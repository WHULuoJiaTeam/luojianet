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
from luojianet_ms import Tensor, Parameter
from luojianet_ms import context
from luojianet_ms.common.api import _CellGraphExecutor
from luojianet_ms.nn import TrainOneStepCell
from luojianet_ms.nn.optim import AdamWeightDecay
from luojianet_ms.ops import operations as P


class NetWithLoss(nn.Cell):
    def __init__(self, network, strategy3):
        super(NetWithLoss, self).__init__()
        self.loss = P.SoftmaxCrossEntropyWithLogits().shard(strategy3)
        self.network = network

    def construct(self, x, b):
        predict = self.network(x)
        return self.loss(predict, b)[0]


def compile_net(net, x, b):
    net.set_auto_parallel()
    _CellGraphExecutor().compile(net, x, b)


def test_optimizer_clone_weight():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, weight):
            super().__init__()
            self.weight = Parameter(weight, "w1")
            self.matmul = P.MatMul(transpose_a=False, transpose_b=True).shard(strategy1)
            self.relu = P.ReLU().shard(strategy2)

        def construct(self, x):
            out = self.matmul(x, self.weight)
            out = self.relu(out)
            return out

    context.set_auto_parallel_context(device_num=4, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    strategy1 = ((2, 1), (2, 1))
    strategy2 = ((4, 1),)
    strategy3 = ((4, 1), (4, 1))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    weight = Tensor(np.ones([64, 32]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    net = Net(strategy1, strategy2, weight)

    optimizer = AdamWeightDecay(net.trainable_params())

    net_with_loss = NetWithLoss(net, strategy3)

    train_net = TrainOneStepCell(net_with_loss, optimizer)

    compile_net(train_net, x, b)


def test_optimizer_clone_weight2():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, weight):
            super().__init__()
            self.weight = Parameter(weight, "w1")
            self.matmul = P.MatMul(transpose_a=False, transpose_b=True).shard(strategy1)
            self.relu = P.ReLU().shard(strategy2)

        def construct(self, x):
            out = self.matmul(x, self.weight)
            out = self.relu(out)
            return out

    context.set_auto_parallel_context(device_num=4, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    strategy1 = ((2, 1), (2, 1))
    strategy2 = ((4, 1),)
    strategy3 = ((4, 1), (4, 1))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    weight = Tensor(np.ones([64, 32]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    net = Net(strategy1, strategy2, weight)

    optimizer = AdamWeightDecay(net.trainable_params())

    net_with_loss = NetWithLoss(net, strategy3)

    train_net = TrainOneStepCell(net_with_loss, optimizer)

    compile_net(train_net, x, b)
