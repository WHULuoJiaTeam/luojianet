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
"""
train Conv2dBnFoldQuant Module
"""

import pytest
import numpy as np
from luojianet_ms import nn
from luojianet_ms import context
from luojianet_ms import Tensor
from luojianet_ms.common import set_seed
from luojianet_ms.compression.quant import create_quant_config

class Net(nn.Module):
    def __init__(self, qconfig):
        super(Net, self).__init__()
        self.conv = nn.Conv2dBnFoldQuant(2, 3, kernel_size=(2, 2), stride=(1, 1),
                                         pad_mode='valid', quant_config=qconfig)
    def forward(self, x):
        return self.conv(x)

def test_conv2d_bn_fold_quant():
    set_seed(1)
    quant_config = create_quant_config()
    network = Net(quant_config)
    inputs = Tensor(np.ones([1, 2, 5, 5]).astype(np.float32))
    label = Tensor(np.ones([1, 3, 4, 4]).astype(np.int32))
    opt = nn.Momentum(filter(lambda x: x.requires_grad, network.get_parameters()), learning_rate=0.1, momentum=0.9)
    loss = nn.MSELoss()
    net_with_loss = nn.WithLossCell(network, loss)
    train_network = nn.TrainOneStepCell(net_with_loss, opt)
    train_network.set_train()
    out_loss = train_network(inputs, label)
    expect_loss = np.array([0.940427])
    error = np.array([0.1])
    diff = out_loss.asnumpy() - expect_loss
    assert np.all(abs(diff) < error)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv2d_bn_fold_quant_ascend():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_conv2d_bn_fold_quant()
