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
import luojianet_ms.nn as nn
from luojianet_ms import Tensor
from luojianet_ms import context
from luojianet_ms.common.api import _cell_graph_executor
from luojianet_ms.ops import composite as C
from luojianet_ms.ops import operations as P
from luojianet_ms.parallel._utils import _reset_op_id as reset_op_id
from tests.ut.python.ops.test_math_ops import VirtualLoss


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Module):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def forward(self, x, y, b):
        predict = self.network(x, y, b)
        return self.loss(predict)


class GradWrap(nn.Module):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def forward(self, x, y, b):
        return grad_all(self.network)(x, y, b)


def test_auto_parallel_l2normalize():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = P.L2Normalize()
            self.norm2 = P.L2Normalize()
            self.mul1 = P.Mul()
            self.mul2 = P.Mul()

        def forward(self, x, y, b):
            y = self.norm1(y)
            x = self.norm2(x)
            out = self.mul1(x, y)
            out = self.mul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = NetWithLoss(Net())
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net.set_auto_parallel()
    reset_op_id()

    x = Tensor(np.ones([128, 64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([128, 64, 64]), dtype=ms.float32)
    b = Tensor(np.ones([128, 64, 64]), dtype=ms.float32)
    net.set_train()
    _cell_graph_executor.compile(net, x, y, b, phase='train')
