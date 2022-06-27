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

import numpy as np

import luojianet_ms as ms
import luojianet_ms.common.dtype as mstype
import luojianet_ms.nn as nn
from luojianet_ms import Tensor
from luojianet_ms import context
from luojianet_ms.common.api import _cell_graph_executor
from luojianet_ms.context import set_auto_parallel_context
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


def compile_net(net, x, y):
    net.set_auto_parallel()
    net.set_train()
    _cell_graph_executor.compile(net, x, y)


# model_parallel test
def test_two_matmul():
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)
            self.matmul3 = P.MatMul().shard(strategy3)
            self.diag = P.Diag()
            self.fill = P.Fill()

        def forward(self, x, y):
            fill = self.diag(self.fill(mstype.float32, (128,), 1.0))
            out1 = self.matmul1(fill, x)
            out2 = self.matmul2(y, fill)
            out = self.matmul3(out1, out2)
            return out

    set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((1, 8), (8, 1))
    strategy3 = ((2, 4), (4, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)

    compile_net(net, x, y)


def test_matmul_mul_broadcast2():
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.mul = P.Mul().shard(strategy2)
            self.t = Tensor(0.9, ms.float32)

        def forward(self, x, y):
            out = self.matmul(x, y)
            out = self.mul(out, self.t)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), ())
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    compile_net(net, x, y)


def test_two_matmul1():
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)
            self.matmul3 = P.MatMul().shard(strategy3)
            self.diag = P.Diag()
            self.fill = P.Fill()

        def forward(self, x, y):
            fill = self.diag(self.fill(mstype.float32, (128,), 1.0))
            out1 = self.matmul1(fill, x)
            out2 = self.matmul2(fill, y)
            out = self.matmul3(out1, out2)
            return out

    set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((1, 8), (8, 1))
    strategy3 = ((2, 4), (4, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 128]), dtype=ms.float32)
    y = Tensor(np.ones([128, 128]), dtype=ms.float32)

    compile_net(net, x, y)


def test_matmul_add_tensor():
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.add = P.Add().shard(strategy2)
            self.b = Tensor(0.9, ms.float32)

        def forward(self, x, y):
            out = self.matmul(x, y)
            out = self.add(out, self.b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), ())
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)

    compile_net(net, x, y)
