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

import re
import numpy as np

import luojianet_ms as ms
import luojianet_ms.nn as nn
from luojianet_ms import Tensor, Parameter
from luojianet_ms import context
from luojianet_ms.common import dtype as mstype
from luojianet_ms.common.api import _cell_graph_executor
from luojianet_ms.ops import operations as P
from luojianet_ms.parallel import set_algo_parameters
from luojianet_ms.parallel._utils import _reset_op_id as reset_op_id
from tests.ut.python.ops.test_math_ops import VirtualLoss


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y):
        predict = self.network(x, y)
        return self.loss(predict)


def test_common_parameter():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul1 = P.MatMul()
            self.matmul2 = P.MatMul()
            self.matmul3 = P.MatMul()
            self.weight1 = Parameter(Tensor(np.ones([64, 64]).astype(np.float16) * 0.01), "w", requires_grad=True)
            self.cast1 = P.Cast()
            self.cast2 = P.Cast()

        def construct(self, x, y):
            m1_result = self.matmul1(x, self.cast1(self.weight1, mstype.float32))
            m2_result = self.matmul2(y, self.cast2(self.weight1, mstype.float32))
            m3_result = self.matmul3(m2_result, m1_result)

            return m3_result

    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0)

    set_algo_parameters(elementwise_op_strategy_follow=True)
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64]), dtype=ms.float32)

    net = NetWithLoss(Net())
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net.set_auto_parallel()
    reset_op_id()

    net.set_train()
    _cell_graph_executor.compile(net, x, y, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('MatMul-op', k) is not None:
            assert v == [[8, 1], [1, 1]]
        elif re.search('Cast-op', k) is not None:
            assert v == [[1, 1]]
