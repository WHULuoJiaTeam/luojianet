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
import luojianet_ms.common.dtype as mstype
import luojianet_ms.nn as nn
from luojianet_ms import Tensor
from luojianet_ms.context import set_auto_parallel_context, ParallelMode
from luojianet_ms.ops import composite as C
from luojianet_ms.parallel.nn import Transformer, TransformerOpParallelConfig, MoEConfig
from luojianet_ms.nn.optim import AdamWeightDecay
from luojianet_ms.nn.wrap.cell_wrapper import TrainOneStepCell, _VirtualDatasetCell
from luojianet_ms.train import Model
from tests.dataset_mock import MindData
from tests.ut.python.ops.test_math_ops import VirtualLoss

grad_all = C.GradOperation(get_all=True)


class Dataset(MindData):
    def __init__(self, *inputs, length=3):
        super(Dataset, self).__init__(size=length)
        self.inputs = inputs
        self.index = 0
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.inputs

    def reset(self):
        self.index = 0


config = TransformerOpParallelConfig(data_parallel=2, model_parallel=8, vocab_emb_dp=False)
moe_config = MoEConfig(expert_num=4)


class NetWithLossFiveInputs(nn.Module):
    def __init__(self, network):
        super(NetWithLossFiveInputs, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def call(self, x1, x2, x3, x4, x5):
        predict, _, _, _ = self.network(x1, x2, x3, x4, x5)
        return self.loss(predict)


def test_transformer_model():
    set_auto_parallel_context(device_num=16, global_rank=0,
                              full_batch=True, enable_alltoall=True,
                              parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    net = Transformer(encoder_layers=1,
                      decoder_layers=1,
                      batch_size=2,
                      src_seq_length=20,
                      tgt_seq_length=10,
                      hidden_size=64,
                      num_heads=8,
                      ffn_hidden_size=64,
                      moe_config=moe_config,
                      parallel_config=config)
    net = _VirtualDatasetCell(net)
    encoder_input_value = Tensor(np.ones((2, 20, 64)), mstype.float32)
    encoder_input_mask = Tensor(np.ones((2, 20, 20)), mstype.float16)
    decoder_input_value = Tensor(np.ones((2, 10, 64)), mstype.float32)
    decoder_input_mask = Tensor(np.ones((2, 10, 10)), mstype.float16)
    memory_mask = Tensor(np.ones((2, 10, 20)), mstype.float16)
    net = NetWithLossFiveInputs(net)
    params = net.trainable_params()
    optimizer = AdamWeightDecay(params)
    dataset = Dataset(encoder_input_value, encoder_input_mask, decoder_input_value, decoder_input_mask,
                      memory_mask)
    net_with_grad = TrainOneStepCell(net, optimizer=optimizer)
    model = Model(net_with_grad)

    model.train(1, dataset, dataset_sink_mode=False)


def test_transformer_model_2d():
    set_auto_parallel_context(device_num=16, global_rank=0,
                              full_batch=True, enable_alltoall=True,
                              parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    net = Transformer(encoder_layers=1,
                      decoder_layers=1,
                      batch_size=2,
                      src_seq_length=20,
                      tgt_seq_length=10,
                      hidden_size=64,
                      num_heads=8,
                      ffn_hidden_size=64,
                      moe_config=moe_config,
                      parallel_config=config)
    net = _VirtualDatasetCell(net)

    encoder_input_value = Tensor(np.ones((40, 64)), mstype.float32)
    encoder_input_mask = Tensor(np.ones((2, 20, 20)), mstype.float16)
    decoder_input_value = Tensor(np.ones((20, 64)), mstype.float32)
    decoder_input_mask = Tensor(np.ones((2, 10, 10)), mstype.float16)
    memory_mask = Tensor(np.ones((2, 10, 20)), mstype.float16)
    net = NetWithLossFiveInputs(net)
    params = net.trainable_params()
    optimizer = AdamWeightDecay(params)
    dataset = Dataset(encoder_input_value, encoder_input_mask, decoder_input_value, decoder_input_mask,
                      memory_mask)
    net_with_grad = TrainOneStepCell(net, optimizer=optimizer)
    model = Model(net_with_grad)

    model.train(1, dataset, dataset_sink_mode=False)
