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

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.parallel._utils import _reset_op_id as reset_op_id
from tests.ut.python.ops.test_math_ops import VirtualLoss


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y, b):
        predict = self.network(x, y, b)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, b):
        return grad_all(self.network)(x, y, b)


# core dump, step_auto_parallel should SetInputs for transpose axis
def test_two_matmul_transpose():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul1 = P.MatMul()
            self.matmul2 = P.MatMul()
            self.transpose1 = P.Transpose()
            self.transpose2 = P.Transpose()

        def construct(self, x, y, b):
            out = self.matmul1(x, y)
            out = self.matmul2(out, b)
            out = self.transpose1(out, (1, 0))
            out = self.transpose2(out, (1, 0))
            return out

    size = 16
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    net = NetWithLoss(Net())
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net.set_auto_parallel()
    reset_op_id()

    net.set_train()
    _cell_graph_executor.compile(net, x, y, b, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    print(strategies)
    expected_strategies = {'Default/network-Net/Transpose-op0': [[1, 16]],
                           'Default/network-Net/Transpose-op1': [[16, 1]],
                           'Default/network-Net/MatMul-op4': [[16, 1], [1, 1]],
                           'Default/network-Net/MatMul-op2': [[16, 1], [1, 1]],
                           'Default/_VirtualDataset-op3': [[16, 1], [16, 1], [16, 1]]}
    assert strategies == expected_strategies
