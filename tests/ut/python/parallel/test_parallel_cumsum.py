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
import pytest

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

    def call(self, x, y):
        predict = self.network(x, y)
        return self.loss(predict)

class GradWrap(nn.Module):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def call(self, x, y):
        return grad_all(self.network)(x, y)

def compile_net(net, x, y):
    net.set_auto_parallel()
    net.set_train()
    _cell_graph_executor.compile(net, x, y)

def test_cumsum_semi():
    """
    Feature: CumSum operatorInfo in parallel.
    Description: MatMul->CumSum
    Expectation: Currently, CumSum does not support the axis dimension split. compile done without error.
    """
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.matmul1 = P.MatMul().shard(((16, 1), (1, 1)))
            self.cumsum = P.CumSum().shard(((16, 1),))

        def call(self, x, y):
            out = self.matmul1(x, y)
            out = self.cumsum(out, 0)
            return out

    size = 16
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)

    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    with pytest.raises(RuntimeError):
        compile_net(net, x, y)


def test_cumsum_semi2():
    """
    Feature: CumSum operatorInfo in parallel.
    Description: MatMul->CumSum
    Expectation: Compile done without error.
    """
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.matmul1 = P.MatMul().shard(((16, 1), (1, 1)))
            self.cumsum = P.CumSum().shard(((1, 16),))

        def call(self, x, y):
            out = self.matmul1(x, y)
            out = self.cumsum(out, 0)
            return out

    size = 16
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)

    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    compile_net(net, x, y)


def test_cumsum_auto():
    """
    Feature: CumSum operatorInfo in parallel.
    Description: MatMul->CumSum
    Expectation: Compile done without error.
    """
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.matmul1 = P.MatMul().shard(((16, 1), (1, 1)))
            self.cumsum = P.CumSum()

        def call(self, x, y):
            out = self.matmul1(x, y)
            out = self.cumsum(out, -1)
            return out

    size = 16
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)

    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    compile_net(net, x, y)
