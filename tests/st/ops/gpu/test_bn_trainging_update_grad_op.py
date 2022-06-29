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
from luojianet_ms.ops.operations import _grad_ops as G
context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


class BNTrainingUpdateGrad(nn.Module):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.bn_training_update_grad = G.BNTrainingUpdateGrad(self.epsilon)

    def forward(self, grads, x, batch_mean, batch_variance):
        diff_scale, diff_offset = self.bn_training_update_grad(grads, x, batch_mean, batch_variance)
        res_list = [diff_scale, diff_offset]
        return res_list


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast_graph():
    """
    Feature: input type is float
    Description: test cases for bn_training_reduce_grad.
    Expectation: print the result.
    """
    grads = Tensor(np.ones([2, 1, 2, 2]), luojianet_ms.float32)
    x = Tensor(np.ones([2, 1, 2, 2]), luojianet_ms.float32)
    batch_mean = Tensor(np.ones([1]), luojianet_ms.float32)
    batch_variance = Tensor(np.ones([1]), luojianet_ms.float32)
    epsilon = float(0.0001)
    bn_training_update_grad = BNTrainingUpdateGrad(epsilon)
    output = bn_training_update_grad(grads, x, batch_mean, batch_variance)
    print(output)
