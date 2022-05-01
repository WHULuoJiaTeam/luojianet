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
import luojianet_ms.ops as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Module):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def call(self, vectors, index):
        predict = self.network(vectors, index)
        return self.loss(predict)


class GradWrap(nn.Module):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def call(self, vectors, index):
        return grad_all(self.network)(vectors, index)


def test_auto_parallel_unsortedsegmentsum():
    class Net(nn.Module):
        def __init__(self, num_segments):
            super().__init__()
            self.merge_op = P.UnsortedSegmentSum()
            self.num_segments = num_segments

        def call(self, vectors, index):
            out = self.merge_op(vectors, index, self.num_segments)
            return out

    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel")

    x = Tensor(np.random.rand(16, 16, 32, 64), dtype=ms.float32)
    indices = Tensor(np.random.randint(16, size=(16, 16)))

    net = GradWrap(NetWithLoss(Net(16)))
    net.set_auto_parallel()
    net.set_train()
    _cell_graph_executor.compile(net, x, indices)
