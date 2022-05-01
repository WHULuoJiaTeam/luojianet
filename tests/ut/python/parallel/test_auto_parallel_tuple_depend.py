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
from luojianet_ms.ops.operations.comm_ops import _VirtualDataset
from tests.ut.python.ops.test_math_ops import VirtualLoss


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Module):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def call(self, x, y, b):
        predict = self.network(x, y, b)
        return self.loss(predict)


class GradWrap(nn.Module):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def call(self, x, y, b):
        return grad_all(self.network)(x, y, b)


def bn_with_initialize(out_channels):
    bn = nn.BatchNorm1d(out_channels, momentum=0.1, eps=1e-5)
    return bn


# model_parallel test
def test_virtual_dataset_3_input():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.virtual_dataset = _VirtualDataset()
            self.matmul1 = P.MatMul()
            self.matmul2 = P.MatMul()
            self.gelu = P.GeLU()
            self.bn1 = bn_with_initialize(2048)

        def call(self, x, y, b):
            x, y, b = self.virtual_dataset(x, y, b)
            out = self.gelu(self.matmul1(x, y))
            b = self.bn1(b)
            out = self.matmul2(out, b)
            return out

    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = GradWrap(NetWithLoss(Net()))
    net.set_auto_parallel()
    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 2048]), dtype=ms.float32)
    net.set_train()
    _cell_graph_executor.compile(net, x, y, b)
