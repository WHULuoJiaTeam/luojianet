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
from luojianet_ms.common.parameter import Parameter
from luojianet_ms.ops import composite as C
from luojianet_ms.ops import operations as P
from luojianet_ms.common.initializer import initializer
from luojianet_ms.nn import TrainOneStepCell, Momentum
from tests.ut.python.ops.test_math_ops import VirtualLoss


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x):
        predict = self.network(x)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x):
        return grad_all(self.network)(x)

def test_unique_column_split():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.unique = P.Unique().shard(((1,),))
            self.relu = P.ReLU()
            self.mul = P.Mul()
            self.embedding_lookp = P.Gather().shard(((1, 8), (1,)))
            self.embedding_table = Parameter(initializer('normal', [2000, 128]),
                                             name='embedding_table')
            self.gatherv2 = P.Gather().shard(((1, 8), (1,)))
            self.reshape = P.Reshape()
            self.matmul = P.MatMul()
            self.mul_weight = Parameter(Tensor(np.full([32, 64, 1], 0.5, dtype=np.float32)), name="mul_weight")

        def construct(self, indices):
            indices_flatten = self.reshape(indices, (-1,))
            unique_id, unique_idx = self.unique(indices_flatten)
            unique_id_weight = self.embedding_lookp(self.embedding_table, unique_id, 0)
            weight_flatten = self.gatherv2(unique_id_weight, unique_idx, 0)
            weight = self.reshape(weight_flatten, (32, 64, 128))
            vx = self.mul(weight, self.mul_weight)
            return vx

    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0, parallel_mode="auto_parallel")
    x = Tensor(np.ones([32, 64]), dtype=ms.int32)
    net = Net()
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_auto_parallel()
    train_net.set_train()
    _cell_graph_executor.compile(train_net, x)

def test_unique_row_split():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.unique = P.Unique().shard(((1,),))
            self.relu = P.ReLU()
            self.mul = P.Mul()
            self.embedding_lookp = P.Gather().shard(((8, 1), (1,)))
            self.embedding_table = Parameter(initializer('normal', [2000, 128]),
                                             name='embedding_table')
            self.gatherv2 = P.Gather().shard(((1, 1), (1,)))
            self.reshape = P.Reshape()
            self.matmul = P.MatMul()
            self.mul_weight = Parameter(Tensor(np.full([32, 64, 1], 0.5, dtype=np.float32)), name="mul_weight")

        def construct(self, indices):
            indices_flatten = self.reshape(indices, (-1,))
            unique_id, unique_idx = self.unique(indices_flatten)
            unique_id_weight = self.embedding_lookp(self.embedding_table, unique_id, 0)
            weight_flatten = self.gatherv2(unique_id_weight, unique_idx, 0)
            weight = self.reshape(weight_flatten, (32, 64, 128))
            vx = self.mul(weight, self.mul_weight)
            return vx

    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0, parallel_mode="semi_auto_parallel")
    x = Tensor(np.ones([32, 64]), dtype=ms.int32)
    net = Net()
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_auto_parallel()
    train_net.set_train()
    _cell_graph_executor.compile(train_net, x)
