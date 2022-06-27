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


class NetSigmoid(nn.Module):
    def __init__(self):
        super(NetSigmoid, self).__init__()
        self.sigmoid = P.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sigmoid():
    x = Tensor(np.array([[[[-1, 1, 10],
                           [1, -1, 1],
                           [10, 1, -1]]]]).astype(np.float32))
    expect = np.array([[[[0.268941, 0.731059, 0.999955],
                         [0.731059, 0.268941, 0.731059],
                         [0.999955, 0.731059, 0.268941]]]]).astype(np.float32)

    error = np.ones(shape=[1, 1, 3, 3]) * 1.0e-6

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    sigmoid = NetSigmoid()
    output = sigmoid(x)
    diff = output.asnumpy() - expect
    assert np.all(abs(diff) < error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    sigmoid = NetSigmoid()
    output = sigmoid(x)
    diff = output.asnumpy() - expect
    assert np.all(abs(diff) < error)
