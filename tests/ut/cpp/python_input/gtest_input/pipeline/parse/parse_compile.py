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
@File  : parse_compile.py
@Author:
@Date  : 2019-03-20
@Desc  : test luojianet_ms compile method
"""
import logging
import numpy as np

import luojianet_ms.nn as nn
from luojianet_ms.common.tensor import Tensor
from luojianet_ms.nn.optim import Momentum
from luojianet_ms.train.model import Model

log = logging.getLogger("test")
log.setLevel(level=logging.ERROR)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal')
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        out = self.flatten(x)
        return out


loss = nn.MSELoss()


def test_build():
    net = Net()
    opt = Momentum(net.get_parameters(), learning_rate=0.1, momentum=0.9)
    Model(net, loss_fn=loss, optimizer=opt, metrics=None)
