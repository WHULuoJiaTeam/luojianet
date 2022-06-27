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
import pytest
import luojianet_ms.context as context
import luojianet_ms.nn as nn
from luojianet_ms import Tensor
from luojianet_ms.ops import operations as P
from luojianet_ms.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class Net(nn.Module):
    def __init__(self, fill_mode=0):
        super(Net, self).__init__()
        self.det_triangle = P.DetTriangle(fill_mode=fill_mode)

    def forward(self, x):
        return self.det_triangle(x)

@pytest.mark.level2
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_1D():
    fill_mode = 0
    input_x = np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]]).astype(np.float32)
    net = Net(fill_mode=fill_mode)
    tx = Tensor(input_x, mstype.float32)
    output = net(tx)
    assert output == 18
