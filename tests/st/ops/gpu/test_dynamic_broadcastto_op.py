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

import luojianet_ms
import luojianet_ms.context as context
import luojianet_ms.nn as nn
import luojianet_ms.ops.operations as ops
from luojianet_ms import Tensor
from luojianet_ms.ops.operations import _inner_ops as inner

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.d_shape = ops.TensorShape()
        self.d_broadcastto = inner.DynamicBroadcastTo()

    def forward(self, data, shape):
        shape = self.d_shape(shape)
        return self.d_broadcastto(data, shape)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_float32():
    """
    Feature: Dynamic BroadcastTo.
    Description: test cases for dynamic_broadcastto.
    Expectation: the result match expected array.
    """
    data = Tensor(np.array([1, 2, 3]), luojianet_ms.float32)
    shape = Tensor(np.zeros((2, 3)), luojianet_ms.int64)
    expect_data = np.array([[1, 2, 3], [1, 2, 3]]).astype(np.float32)
    net = Net()
    output = net(data, shape)
    print(output.asnumpy())
    assert np.array_equal(output.asnumpy(), expect_data)
