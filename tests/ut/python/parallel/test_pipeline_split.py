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
from luojianet_ms import context
from luojianet_ms import Tensor
from luojianet_ms.ops import operations as P
from luojianet_ms.common.parameter import Parameter
from luojianet_ms.common.initializer import initializer
from luojianet_ms.train.model import Model
from luojianet_ms.nn.wrap.cell_wrapper import PipelineCell, MicroBatchInterleaved


class DatasetLenet():
    def __init__(self, data, label, length=3):
        self.data = data
        self.label = label
        self.index = 1
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.data, self.label

    def reset(self):
        self.index = 0

    def get_dataset_size(self):
        return 32

    def get_repeat_count(self):
        return 1

    def get_batch_size(self):
        return 32

    def create_tuple_iterator(self, num_epochs=1, do_copy=True):
        return self


class MatMulCell(nn.Module):
    def __init__(self, strategy1, strategy2, param=None):
        super().__init__()
        self.param = Parameter(initializer("zeros", [64, 64]), name="param")
        if param is not None:
            self.param = param
        self.param1 = Parameter(initializer("zeros", [64, 64]), name="param1")
        self.matmul = P.MatMul().shard(strategy1)
        self.matmul1 = P.MatMul().shard(strategy2)

    def call(self, x):
        out = self.matmul(x, self.param)
        out = self.matmul1(out, self.param1)
        return out


class Net(nn.Module):
    def __init__(self, strategy1, strategy2, param=None):
        super().__init__()
        self.block = nn.CellList()
        for i in range(2):
            cell = MatMulCell(strategy1, strategy2, param)
            cell.pipeline_stage = i
            self.block.append(cell)

    def call(self, x):
        for i in range(2):
            x = self.block[i](x)
        return x


class PipelineSplit(nn.Module):
    def __init__(self, strategy1, strategy2):
        super().__init__()
        self.cell = Net(strategy1, strategy2)
        self.cell.block[0].matmul.add_prim_attr("parameter_start", 0)

    def call(self, x, label):
        x = self.cell(x)
        return x


class PipelineSplit2(nn.Module):
    def __init__(self, strategy1, strategy2):
        super().__init__()
        self.param = Parameter(initializer("zeros", [64, 64]), name="param")
        self.cell = Net(strategy1, strategy2, self.param)
        self.cell.block[0].matmul.add_prim_attr("parameter_start", 0)

    def call(self, x, label):
        x = self.cell(x)
        return x


def test_pipeline_split_stage0():
    context.set_auto_parallel_context(device_num=8, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((4, 1), (1, 1))
    strategy2 = ((2, 1), (1, 1))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 4)
    params = net.network.cell.block[0].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
    for _, param in model._train_network.parameters_and_names():
        assert param.name != "cell.block.1.param"
        assert param.name != "cell.block.1.param1"

def test_pipeline_split_stage1():
    context.set_auto_parallel_context(device_num=8, global_rank=4, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((4, 1), (1, 1))
    strategy2 = ((2, 1), (1, 1))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 4)
    params = net.network.cell.block[1].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
    for _, param in model._train_network.parameters_and_names():
        assert param.name != "cell.block.0.param"
        assert param.name != "cell.block.0.param1"


def test_pipeline_split_shared_parameter_stage0():
    context.set_auto_parallel_context(device_num=8, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((4, 1), (1, 1))
    strategy2 = ((2, 1), (1, 1))
    net = PipelineCell(PipelineSplit2(strategy1, strategy2), 4)
    params = net.network.cell.block[0].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_shared_parameter_stage1():
    context.set_auto_parallel_context(device_num=8, global_rank=4, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((4, 1), (1, 1))
    strategy2 = ((2, 1), (1, 1))
    net = PipelineCell(PipelineSplit2(strategy1, strategy2), 4)
    params = net.network.cell.block[1].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_shared_parameter_stage0_predict():
    context.set_auto_parallel_context(device_num=8, global_rank=0, pipeline_stages=2, full_batch=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((4, 1), (1, 1))
    strategy2 = ((2, 1), (1, 1))
    net = PipelineSplit2(strategy1, strategy2)
    model = Model(net)
    model.predict(data, label)


def test_pipeline_split_shared_parameter_stage1_predict():
    context.set_auto_parallel_context(device_num=8, global_rank=4, pipeline_stages=2, full_batch=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((4, 1), (1, 1))
    strategy2 = ((2, 1), (1, 1))
    net = PipelineSplit2(strategy1, strategy2)
    model = Model(net)
    model.predict(data, label)


def test_pipeline_split_stage0_opt_shard():
    context.set_auto_parallel_context(device_num=8, global_rank=0, pipeline_stages=2, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((4, 1), (1, 1))
    strategy2 = ((2, 1), (1, 1))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 4)
    params = net.network.cell.block[0].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
    for _, param in model._train_network.parameters_and_names():
        assert param.name != "cell.block.1.param"
        assert param.name != "cell.block.1.param1"


def test_pipeline_split_stage1_opt_shard():
    context.set_auto_parallel_context(device_num=8, global_rank=4, pipeline_stages=2, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((4, 1), (1, 1))
    strategy2 = ((2, 1), (1, 1))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 4)
    params = net.network.cell.block[1].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
    for _, param in model._train_network.parameters_and_names():
        assert param.name != "cell.block.0.param"
        assert param.name != "cell.block.0.param1"


def test_pipeline_split_shared_parameter_stage0_opt_shard():
    context.set_auto_parallel_context(device_num=8, global_rank=0, pipeline_stages=2, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((4, 1), (1, 1))
    strategy2 = ((2, 1), (1, 1))
    net = PipelineCell(PipelineSplit2(strategy1, strategy2), 4)
    params = net.network.cell.block[0].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_shared_parameter_stage1_opt_shard():
    context.set_auto_parallel_context(device_num=8, global_rank=4, pipeline_stages=2, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((4, 1), (1, 1))
    strategy2 = ((2, 1), (1, 1))
    net = PipelineCell(PipelineSplit2(strategy1, strategy2), 4)
    params = net.network.cell.block[1].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_with_micro_batch_interleaved_stage0():
    """
    Feature: test PipelineSplit with MicroBatchInterleaved in auto parallel.
    Description: net with MicroBatchInterleaved in semi auto parallel.
    Expectation: success.
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((4, 1), (1, 1))
    strategy2 = ((2, 1), (1, 1))
    micro_batch_interleaved = 2
    net = PipelineCell(MicroBatchInterleaved(PipelineSplit(strategy1, strategy2), micro_batch_interleaved), 4)
    params = net.network.network.cell.block[0].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
    for _, param in model._train_network.parameters_and_names():
        assert param.name != "cell.block.1.param"
        assert param.name != "cell.block.1.param1"


def test_pipeline_split_with_micro_batch_interleaved_stage1():
    """
    Feature: test PipelineSplit with MicroBatchInterleaved in auto parallel.
    Description: net with MicroBatchInterleaved in semi auto parallel.
    Expectation: success.
    """
    context.set_auto_parallel_context(device_num=8, global_rank=4, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((4, 1), (1, 1))
    strategy2 = ((2, 1), (1, 1))
    micro_batch_interleaved = 2
    net = PipelineCell(MicroBatchInterleaved(PipelineSplit(strategy1, strategy2), micro_batch_interleaved), 4)
    params = net.network.network.cell.block[1].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
    for _, param in model._train_network.parameters_and_names():
        assert param.name != "cell.block.0.param"
        assert param.name != "cell.block.0.param1"


def test_pipeline_split_shared_parameter_with_micro_batch_interleaved_stage0_opt_shard():
    """
    Feature: test PipelineSplitSharedParameter with MicroBatchInterleaved in auto parallel.
    Description: net with MicroBatchInterleaved in semi auto parallel.
    Expectation: success.
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, pipeline_stages=2, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((4, 1), (1, 1))
    strategy2 = ((2, 1), (1, 1))
    micro_batch_interleaved = 2
    net = PipelineCell(MicroBatchInterleaved(PipelineSplit2(strategy1, strategy2), micro_batch_interleaved), 4)
    params = net.network.network.cell.block[0].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_shared_parameter_with_micro_batch_interleaved_stage1_opt_shard():
    """
    Feature: test PipelineSplitSharedParameter with MicroBatchInterleaved in auto parallel.
    Description: net with MicroBatchInterleaved in semi auto parallel.
    Expectation: success.
    """
    context.set_auto_parallel_context(device_num=8, global_rank=4, pipeline_stages=2, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((4, 1), (1, 1))
    strategy2 = ((2, 1), (1, 1))
    micro_batch_interleaved = 2
    net = PipelineCell(MicroBatchInterleaved(PipelineSplit2(strategy1, strategy2), micro_batch_interleaved), 4)
    params = net.network.network.cell.block[1].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
