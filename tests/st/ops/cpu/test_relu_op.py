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
from luojianet_ms.common.initializer import initializer
from luojianet_ms.common.parameter import Parameter
from luojianet_ms.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class NetRelu(nn.Module):
    def __init__(self):
        super(NetRelu, self).__init__()
        self.relu = P.ReLU()
        self.x = Parameter(initializer(Tensor(np.array([[[[-1, 1, 10],
                                                          [1, -1, 1],
                                                          [10, 1, -1]]]]).astype(np.float32)), [1, 1, 3, 3]), name='x')

    def call(self):
        return self.relu(self.x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_relu():
    relu = NetRelu()
    output = relu()
    expect = np.array([[[[0, 1, 10,],
                         [1, 0, 1,],
                         [10, 1, 0.]]]]).astype(np.float32)
    print(output)
    assert (output.asnumpy() == expect).all()
