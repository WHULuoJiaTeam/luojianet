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
""" Test L1Regularizer """
import numpy as np
import pytest
import luojianet_ms.nn as nn
import luojianet_ms.context as context
from luojianet_ms import Tensor, ms_function

context.set_context(mode=context.GRAPH_MODE)


class Net_l1_regularizer(nn.Module):
    def __init__(self, scale):
        super(Net_l1_regularizer, self).__init__()
        self.l1_regularizer = nn.L1Regularizer(scale)

    @ms_function
    def forward(self, weights):
        return self.l1_regularizer(weights)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_l1_regularizer01():
    scale = 0.5
    weights = Tensor(np.array([[1.0, -2.0], [-3.0, 4.0]]).astype(np.float32))
    l1_regularizer = Net_l1_regularizer(scale)
    output = l1_regularizer(weights)
    print("After l1_regularizer01 is: ", output.asnumpy())
    print("output.shape: ", output.shape)
    print("output.dtype: ", output.dtype)
    expect = 5.0
    assert np.all(output.asnumpy() == expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_l1_regularizer08():
    scale = 0.5
    net = nn.L1Regularizer(scale)
    weights = Tensor(np.array([[1.0, -2.0], [-3.0, 4.0]]).astype(np.float32))
    output = net(weights)
    expect = 5.0
    print("output : ", output.asnumpy())
    assert np.all(output.asnumpy() == expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_l1_regularizer_input_int():
    scale = 0.5
    net = nn.L1Regularizer(scale)
    weights = 2
    try:
        output = net(weights)
        print("output : ", output.asnumpy())
    except (TypeError, RuntimeError):
        assert True


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_l1_regularizer_input_tuple():
    scale = 0.5
    net = nn.L1Regularizer(scale)
    weights = (1, 2, 3, 4)
    try:
        output = net(weights)
        print("output : ", output.asnumpy())
    except (TypeError, RuntimeError):
        assert True
