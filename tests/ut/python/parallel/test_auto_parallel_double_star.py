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
from tests.ut.python.ops.test_math_ops import VirtualLoss


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Module):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def forward(self, x, y, z, w, a, b, c):
        predict = self.network(x, y, z, w, a, b, c)
        return self.loss(predict)


class GradWrap(nn.Module):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def forward(self, x, y, z, w, a, b, c):
        return grad_all(self.network)(x, y, z, w, a, b, c)

    # model_parallel test


def test_double_star_graph():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.matmul1 = P.MatMul()
            self.matmul2 = P.MatMul()
            self.matmul3 = P.MatMul()
            self.matmul4 = P.MatMul()
            self.matmul5 = P.MatMul()
            self.matmul6 = P.MatMul()

        def forward(self, x, y, z, w, a, b, c):
            m1_result = self.matmul1(x, y)
            m2_result = self.matmul2(z, w)
            m3_result = self.matmul3(m2_result, m1_result)
            m4_result = self.matmul4(a, b)
            m5_result = self.matmul5(m3_result, m4_result)
            out = self.matmul6(m5_result, c)

            return out

    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    x = Tensor(np.ones([32, 8]), dtype=ms.float32)
    y = Tensor(np.ones([8, 16]), dtype=ms.float32)
    z = Tensor(np.ones([8, 16]), dtype=ms.float32)
    w = Tensor(np.ones([16, 32]), dtype=ms.float32)
    a = Tensor(np.ones([16, 8]), dtype=ms.float32)
    b = Tensor(np.ones([8, 32]), dtype=ms.float32)
    c = Tensor(np.ones([32, 32]), dtype=ms.float32)

    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net.set_auto_parallel()
    net.set_train()
    _cell_graph_executor.compile(net, x, y, z, w, a, b, c)
