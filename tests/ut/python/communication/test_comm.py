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

""" test Communicate """
import numpy as np

import luojianet_ms.context as context
import luojianet_ms.nn as nn
from luojianet_ms import Tensor
from luojianet_ms.common.api import _cell_graph_executor
from luojianet_ms.communication._comm_helper import Backend
from luojianet_ms.communication.management import HCCL_WORLD_COMM_GROUP, NCCL_WORLD_COMM_GROUP, GlobalComm, init
from luojianet_ms.nn import Dense
from luojianet_ms.nn import Momentum
from luojianet_ms.nn import ReLU
from luojianet_ms.nn import TrainOneStepCell, WithLossCell
from luojianet_ms.ops.operations.comm_ops import AllReduce, AllGather, AlltoAll, ReduceOp, ReduceScatter
from luojianet_ms.ops.operations.comm_ops import Broadcast, _AllSwap
from luojianet_ms.ops.operations.array_ops import Gather
import luojianet_ms


# pylint: disable=W0212
# W0212: protected-access

tag = 0

context.set_context(device_target="Ascend")
GlobalComm.CHECK_ENVS = False
init("hccl")
GlobalComm.CHECK_ENVS = True


class AllReduceNet(nn.Module):
    """AllReduceNet definition"""

    def __init__(self, input_channel, out_channel, op):
        super(AllReduceNet, self).__init__()
        self.dense = Dense(input_channel, out_channel)
        self.reduce = AllReduce(op)
        self.relu = ReLU()

    def forward(self, x):
        x = self.dense(x)
        x = self.reduce(x)
        return self.relu(x)


class BroadCastNet(nn.Module):
    """BroadCastNet definition"""

    def __init__(self, input_channel, out_channel):
        super(BroadCastNet, self).__init__()
        self.dense = Dense(input_channel, out_channel)
        self.broadcast = Broadcast(0)

    def forward(self, x):
        x, = self.broadcast((x,))
        x = self.dense(x)
        return x


class AllGatherNet(nn.Module):
    """AllGatherNet definition"""

    def __init__(self, input_channel, out_channel):
        super(AllGatherNet, self).__init__()
        self.dense = Dense(input_channel, out_channel)
        if GlobalComm.BACKEND is Backend.HCCL:
            self.allgather = AllGather(group=HCCL_WORLD_COMM_GROUP)
        elif GlobalComm.BACKEND is Backend.NCCL:
            self.allgather = AllGather(group=NCCL_WORLD_COMM_GROUP)
        else:
            self.allgather = AllGather()

        self.relu = ReLU()

    def forward(self, x):
        x = self.dense(x)
        x = self.allgather(x)
        return self.relu(x)


class ReduceScatterNet(nn.Module):
    """ReduceScatterNet definition"""

    def __init__(self, input_channel, out_channel, op):
        super(ReduceScatterNet, self).__init__()
        self.dense = Dense(input_channel, out_channel)
        self.reducescatter = ReduceScatter(op)
        self.relu = ReLU()

    def forward(self, x):
        x = self.dense(x)
        x = self.reducescatter(x)
        return self.relu(x)


class AlltoAllNet(nn.Module):
    """AlltoAllNet definition"""

    def __init__(self, input_channel, out_channel):
        super(AlltoAllNet, self).__init__()
        self.dense = Dense(input_channel, out_channel)
        self.alltoall = AlltoAll(1, 0, 1)
        self.relu = ReLU()

    def forward(self, x):
        x = self.dense(x)
        x = self.alltoall(x)
        return self.relu(x)


class AllSwapNet(nn.Module):
    """AlltoAllNet definition"""

    def __init__(self, batch_size, input_channel, out_channel):
        super(AllSwapNet, self).__init__()
        self.dense = Dense(input_channel, out_channel)
        self.allswap = _AllSwap()
        self.relu = ReLU()
        part_slice = batch_size / 2
        self.send_size = Tensor([0, part_slice*out_channel, part_slice*out_channel], luojianet_ms.int64)
        self.recv_size = Tensor([part_slice*out_channel, part_slice*out_channel, 0], luojianet_ms.int64)
        self.gatherv2 = Gather()
        self.input = Tensor(np.ones([1]), luojianet_ms.int32)
    def forward(self, x):
        x = self.allswap(x, self.send_size, self.recv_size)
        x = self.relu(x)
        x = self.gatherv2(x, self.input, 0)
        return x


def run_allreduce(op):
    """run_allreduce"""
    context.set_context(mode=context.GRAPH_MODE)
    input_tensor = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]], dtype=np.float32))
    label_tensor = Tensor(np.array([[1.2], [2.2]], dtype=np.float32))
    network = AllReduceNet(2, 1, op)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Momentum(filter(lambda x: x.requires_grad, network.get_parameters()),
                         learning_rate=0.1,
                         momentum=0.9)
    network = WithLossCell(network, loss_fn)
    network = TrainOneStepCell(network, optimizer)
    _cell_graph_executor.compile(network, input_tensor, label_tensor)


def test_allreduce():
    """test_allreduce"""
    context.set_context(mode=context.GRAPH_MODE)
    run_allreduce(ReduceOp.SUM)
    run_allreduce(ReduceOp.MAX)
    run_allreduce(ReduceOp.MIN)
    run_allreduce(ReduceOp.PROD)


def test_allgather():
    """test_allgather"""
    context.set_context(mode=context.GRAPH_MODE)
    input_tensor = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]], dtype=np.float32))
    label_tensor = Tensor(np.array([[1.2], [2.2]], dtype=np.float32))
    network = AllGatherNet(2, 1)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Momentum(filter(lambda x: x.requires_grad, network.get_parameters()),
                         learning_rate=0.1,
                         momentum=0.9)
    network = WithLossCell(network, loss_fn)
    network = TrainOneStepCell(network, optimizer)
    _cell_graph_executor.compile(network, input_tensor, label_tensor)

def test_allswap():
    """run_allswap"""
    context.set_context(mode=context.GRAPH_MODE)
    input_tensor = Tensor(np.ones((100, 20)), dtype=luojianet_ms.float32)
    label_tensor = Tensor(np.ones((1, 20)), dtype=luojianet_ms.float32)
    network = AllSwapNet(100, 20, 20)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Momentum(filter(lambda x: x.requires_grad, network.get_parameters()),
                         learning_rate=0.1,
                         momentum=0.9)
    network = WithLossCell(network, loss_fn)
    network = TrainOneStepCell(network, optimizer)
    _cell_graph_executor.compile(network, input_tensor, label_tensor)


def run_reducescatter(op):
    """run_reducescatter"""
    context.set_context(mode=context.GRAPH_MODE)
    input_tensor = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]], dtype=np.float32))
    label_tensor = Tensor(np.array([[1.2], [2.2]], dtype=np.float32))
    network = ReduceScatterNet(2, 1, op)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Momentum(filter(lambda x: x.requires_grad, network.get_parameters()),
                         learning_rate=0.1,
                         momentum=0.9)
    network = WithLossCell(network, loss_fn)
    network = TrainOneStepCell(network, optimizer)
    _cell_graph_executor.compile(network, input_tensor, label_tensor)


def test_reducescatter():
    """test_reducescatter"""
    context.set_context(mode=context.GRAPH_MODE)
    run_reducescatter(ReduceOp.SUM)


def test_broadcast():
    """test_broadcast"""
    context.set_context(mode=context.GRAPH_MODE)
    input_tensor_1 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]], dtype=np.float32))
    label_tensor = Tensor(np.array([[1.2], [2.2]], dtype=np.float32))
    network = BroadCastNet(2, 1)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Momentum(filter(lambda x: x.requires_grad, network.get_parameters()),
                         learning_rate=0.1,
                         momentum=0.9)
    network = WithLossCell(network, loss_fn)
    network = TrainOneStepCell(network, optimizer)
    _cell_graph_executor.compile(network, input_tensor_1, label_tensor)


def test_alltoall():
    """test_alltoall"""
    context.set_context(mode=context.GRAPH_MODE)
    input_tensor = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]], dtype=np.float32))
    label_tensor = Tensor(np.array([[1.2], [2.2]], dtype=np.float32))
    network = AlltoAllNet(2, 1)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Momentum(filter(lambda x: x.requires_grad, network.get_parameters()),
                         learning_rate=0.1,
                         momentum=0.9)
    network = WithLossCell(network, loss_fn)
    network = TrainOneStepCell(network, optimizer)
    _cell_graph_executor.compile(network, input_tensor, label_tensor)
