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

import luojianet_ms.context as context
import luojianet_ms.nn as nn
from luojianet_ms import Tensor
from luojianet_ms.common.api import ms_function
from luojianet_ms.ops import operations as P

context.set_context(device_target="Ascend")


class Net(nn.Module):
    def __init__(self, is_grad=False):
        super(Net, self).__init__()
        self.SparseSoftmaxCrossEntropyWithLogits = P.SparseSoftmaxCrossEntropyWithLogits(is_grad=is_grad)

    @ms_function
    def call(self, features, labels):
        return self.SparseSoftmaxCrossEntropyWithLogits(features, labels)


def test_net():
    features = np.random.randn(32, 1001, 1, 1).astype(np.float16)
    labels = np.random.randn(32, 1, 1, 1).astype(np.int32)
    gradSparseSoftmaxCrossEntropyWithLogits = Net(is_grad=True)
    gradOutput = gradSparseSoftmaxCrossEntropyWithLogits(Tensor(features), Tensor(labels))
    print(gradOutput[0].asnumpy())
