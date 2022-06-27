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
"""Test case."""
import numpy as np

import luojianet_ms
import luojianet_ms.nn as nn
from luojianet_ms import Tensor, context


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.conv2 = nn.Conv2d(1, 3, 5, has_bias=True)
        self.layers = (self.conv1, self.conv2)

    def forward(self, x, index):
        x = self.layers[index](x)
        return 2 + x


def test_case():
    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    data = Tensor(np.ones((1, 1, 224, 224)), luojianet_ms.float32)
    idx = Tensor(1, luojianet_ms.int32)
    net(data, idx)
