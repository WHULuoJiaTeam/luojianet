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
import luojianet_ms.nn as nn
import luojianet_ms.context as context
from luojianet_ms import Tensor
from luojianet_ms.ops import operations as P
context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


class BNTrainingUpdate(nn.Module):
    def __init__(self, is_ref, epsilon, factor):
        super().__init__()
        self.epsilon = epsilon
        self.factor = factor
        self.is_ref = is_ref
        self.bn_training_update = P.BNTrainingUpdate(self.is_ref, self.epsilon, self.factor)

    def forward(self, x, input_sum, square_sum, scale, offset, mean, variance):
        y, mean_out, variance_out, batch_mean, batch_variance, = self.bn_training_update(x, input_sum, square_sum,
                                                                                         scale, offset, mean, variance)
        res = [y, mean_out, variance_out, batch_mean, batch_variance]
        return res


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast_graph():
    """
    Feature: input type is float
    Description: test cases for bn_training_reduce_grad.
    Expectation: print the result.
    """
    x = Tensor(np.ones([2, 1, 2, 2]), luojianet_ms.float32)
    input_sum = Tensor(np.ones([1]), luojianet_ms.float32)
    square_sum = Tensor(np.ones([1]), luojianet_ms.float32)
    scale = Tensor(np.ones([1]), luojianet_ms.float32)
    offset = Tensor(np.ones([1]), luojianet_ms.float32)
    mean = Tensor(np.ones([1]), luojianet_ms.float32)
    variance = Tensor(np.ones([1]), luojianet_ms.float32)
    epsilon = float(0.0001)
    factor = 0.1
    bn_training_update = BNTrainingUpdate(is_ref=True, epsilon=epsilon, factor=factor)
    output = bn_training_update(x, input_sum, square_sum, scale, offset, mean, variance)
    print(output)
