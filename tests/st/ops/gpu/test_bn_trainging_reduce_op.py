# Copyright 2019-2021 Huawei Technologies Co., Ltd# Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
# Copyright 2021, 2022 Huawei Technologies Co., Ltd#
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

import luojianet_ms.nn as nn
import luojianet_ms.context as context
from luojianet_ms import Tensor
from luojianet_ms.ops import operations as P
context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


class BNTrainingReduce(nn.Module):
    def __init__(self):
        super(BNTrainingReduce, self).__init__()
        self.bn_training_reduce = P.BNTrainingReduce()

    def forward(self, x):
        return self.bn_training_reduce(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast_graph():
    """
    Feature: input type is float
    Description: test cases for bn_training_reduce.
    Expectation: print the result.
    """
    x = Tensor(np.random.rand(2, 3, 4, 4).astype(np.float32)).astype(np.float32)
    bn_training_reduce = BNTrainingReduce()
    out_sum, out_square_sum = bn_training_reduce(x)
    print(out_sum)
    print(out_square_sum)
