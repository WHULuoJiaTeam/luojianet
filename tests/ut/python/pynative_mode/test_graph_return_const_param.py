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
""" test_return_const_or_parameter """

import numpy as np

import luojianet_ms.common.dtype as mstype
import luojianet_ms.nn as nn
from luojianet_ms import context
from luojianet_ms.common.api import ms_function
from luojianet_ms.common.parameter import Parameter
from luojianet_ms.common.tensor import Tensor


def setup_module(module):
    context.set_context(mode=context.PYNATIVE_MODE)


class ChooseInitParameter(nn.Module):
    def __init__(self):
        super(ChooseInitParameter, self).__init__()
        self.x = Parameter(Tensor(np.ones(2), dtype=mstype.int32), name='x')

    @ms_function
    def call(self):
        return self.x


class ChooseInitParameterWithInput(nn.Module):
    def __init__(self):
        super(ChooseInitParameterWithInput, self).__init__()
        self.x = Parameter(Tensor(np.ones(2), dtype=mstype.int32), name='x')

    @ms_function
    def call(self, input_data):
        return self.x


def test_choose_init_param():
    choose = ChooseInitParameter()
    expect = Tensor(np.ones(2), dtype=mstype.int32)
    out = choose()
    assert np.allclose(out.asnumpy(), expect.asnumpy())


def test_choose_param_with_input():
    choose = ChooseInitParameterWithInput()
    input_data = Tensor(np.zeros(2), dtype=mstype.int32)
    expect = Tensor(np.ones(2), dtype=mstype.int32)
    out = choose(input_data)
    assert np.allclose(expect.asnumpy(), out.asnumpy())
