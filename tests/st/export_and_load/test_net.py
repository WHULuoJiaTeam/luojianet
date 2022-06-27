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
"""Export net test."""
import os
import numpy as np
import pytest

import luojianet_ms as ms
import luojianet_ms.nn as nn
from luojianet_ms import context
from luojianet_ms.common.tensor import Tensor
from luojianet_ms.train.serialization import export


class SliceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x, y):
        x = self.relu(x)
        x[2,] = y
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_export_slice_net():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    input_x = Tensor(np.random.rand(4, 4, 4), ms.float32)
    input_y = Tensor(np.array([1]), ms.float32)
    net = SliceNet()
    file_name = "slice_net"
    export(net, input_x, input_y, file_name=file_name, file_format='AIR')
    verify_name = file_name + ".air"
    assert os.path.exists(verify_name)
    os.remove(verify_name)
    export(net, input_x, input_y, file_name=file_name, file_format='MINDIR')

    verify_name = file_name + ".mindir"
    assert os.path.exists(verify_name)
    os.remove(verify_name)
