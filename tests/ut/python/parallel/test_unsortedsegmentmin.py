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
# ============================================================================

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

context.set_context(mode=context.GRAPH_MODE)


grad_all = C.GradOperation(get_all=True)


class Net(nn.Module):
    def __init__(self, strategy1, strategy2, num_segments):
        super(Net, self).__init__()
        self.virtual_dataset = _VirtualDataset()
        self.merge_op = P.UnsortedSegmentMin().shard((strategy1, strategy2))
        self.num_segments = num_segments

    def call(self, vectors, segment_ids):
        predict = self.merge_op(vectors, segment_ids, self.num_segments)
        return predict


class GradWrap(nn.Module):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def call(self, x, y):
        return grad_all(self.network)(x, y)


class NetWithLoss(nn.Module):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def call(self, x, y):
        predict = self.network(x, y)
        return self.loss(predict)


def compile_graph(x, y, segments, strategy1, strategy2, auto=False):
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, segments)))
    net.set_auto_parallel()
    net.set_train()
    if auto:
        context.set_auto_parallel_context(parallel_mode="auto_parallel")
    else:
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    _cell_graph_executor.compile(net, x, y)


def test_unsortedsegmentmin_model_parallel_slice_1d():
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    x = Tensor(np.ones(8), ms.float32)
    y = Tensor(np.ones(8), ms.int32)
    num_segments = 16
    strategy1 = (8,)
    strategy2 = (8,)
    compile_graph(x, y, num_segments, strategy1, strategy2)


def test_unsortedsegmentmin_model_parallel_no_slice_1d():
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    x = Tensor(np.ones(8), ms.float32)
    y = Tensor(np.ones(8), ms.int32)
    num_segments = 16
    strategy1 = (1,)
    strategy2 = (1,)
    compile_graph(x, y, num_segments, strategy1, strategy2)


def test_unsortedsegmentmin_model_parallel_index_slice_2d():
    context.set_auto_parallel_context(device_num=4, global_rank=0)
    x = Tensor(np.ones((4, 8)), ms.float32)
    y = Tensor(np.arange(4), ms.int32)
    num_segments = 4
    strategy1 = (4, 1)
    strategy2 = (4,)
    compile_graph(x, y, num_segments, strategy1, strategy2)


def test_unsortedsegmentmin_model_parallel_vector_slice_2d():
    context.set_auto_parallel_context(device_num=4, global_rank=0)
    x = Tensor(np.ones((4, 8)), ms.float32)
    y = Tensor(np.ones(4), ms.int32)
    num_segments = 4
    strategy1 = (1, 4)
    strategy2 = (1,)
    compile_graph(x, y, num_segments, strategy1, strategy2)


def test_unsortedsegmentmin_model_parallel_vector_slice_3d():
    context.set_auto_parallel_context(device_num=4, global_rank=0)
    x = Tensor(np.ones((4, 8, 8)), ms.float32)
    y = Tensor(np.ones(4), ms.int32)
    num_segments = 4
    strategy1 = (1, 2, 2)
    strategy2 = (1,)
    compile_graph(x, y, num_segments, strategy1, strategy2)


def test_unsortedsegmentmin_model_parallel_index_vector_slice_2d():
    context.set_auto_parallel_context(device_num=4, global_rank=0)
    x = Tensor(np.ones((4, 8)), ms.float32)
    y = Tensor(np.ones(4), ms.int32)
    num_segments = 4
    strategy1 = (2, 2)
    strategy2 = (2,)
    compile_graph(x, y, num_segments, strategy1, strategy2)


def test_unsortedsegmentmin_model_parallel_index_vector_slice_3d():
    context.set_auto_parallel_context(device_num=4, global_rank=0)
    x = Tensor(np.ones((4, 4, 8)), ms.float32)
    y = Tensor(np.ones((4)), ms.int32)
    num_segments = 16
    strategy1 = (2, 1, 2)
    strategy2 = (2,)
    compile_graph(x, y, num_segments, strategy1, strategy2)

def test_unsortedsegmentmin_model_parallel_float16():
    context.set_auto_parallel_context(device_num=4, global_rank=0)
    x = Tensor(np.ones((4, 4, 8)), ms.float16)
    y = Tensor(np.ones((4)), ms.int32)
    num_segments = 16
    strategy1 = (2, 1, 2)
    strategy2 = (2,)
    compile_graph(x, y, num_segments, strategy1, strategy2)

def test_unsortedsegmentmin_model_parallel_int32():
    context.set_auto_parallel_context(device_num=4, global_rank=0)
    x = Tensor(np.ones((4, 4, 8)), ms.int32)
    y = Tensor(np.ones((4)), ms.int32)
    num_segments = 16
    strategy1 = (2, 1, 2)
    strategy2 = (2,)
    compile_graph(x, y, num_segments, strategy1, strategy2)
