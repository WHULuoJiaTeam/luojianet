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


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Module):
    def __init__(self, network, strategy3):
        super(NetWithLoss, self).__init__()
        self.loss = P.SoftmaxCrossEntropyWithLogits().shard(strategy3)
        self.network = network

    def call(self, x, y, bias, label):
        predict = self.network(x, y, bias)
        return self.loss(predict, label)[0]


class GradWrap(nn.Module):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def call(self, x, y, bias, label):
        return grad_all(self.network)(x, y, bias, label)


def test_linear():
    class Net(nn.Module):
        def __init__(self, strategy0, strategy1, strategy2):
            super().__init__()
            self.fc_nobias = P.MatMul(transpose_b=True).shard(strategy0)
            self.add = P.Add().shard(strategy1)
            self.gelu = P.GeLU().shard(strategy2)

        def call(self, x, y, bias):
            out = self.fc_nobias(x, y)
            out = self.add(out, bias)
            out = self.gelu(out)
            return out

    context.set_auto_parallel_context(device_num=16, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy0 = ((2, 4), (2, 4))
    strategy1 = ((2, 4), (4,))
    strategy2 = ((2, 8),)
    strategy3 = ((16, 1), (16, 1))
    net = GradWrap(NetWithLoss(Net(strategy0, strategy1, strategy2), strategy3))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 32]), dtype=ms.float32)
    bias = Tensor(np.ones([64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    net.set_train()
    _cell_graph_executor.compile(net, x, y, bias, label)
