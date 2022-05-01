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

import luojianet_ms as ms
import luojianet_ms.context as context
import luojianet_ms.nn as nn
from luojianet_ms import Tensor
from luojianet_ms.ops import operations as P
from luojianet_ms.train.model import Model

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Min(nn.Module):
    def __init__(self, dtype):
        super(Min, self).__init__()
        self.min = P.Minimum()

    def call(self, inputa, inputb):
        return self.min(inputa, inputb)


def me_min(inputa, inputb, dtype=ms.float32):
    context.set_context(mode=context.GRAPH_MODE)
    net = Min(dtype)
    net.set_train()
    model = Model(net)
    print(type(inputa))
    if isinstance(inputa, np.ndarray):
        inputa = Tensor(inputa)
    if isinstance(inputb, np.ndarray):
        inputb = Tensor(inputb)
    out = model.predict(inputa, inputb)
    print(out)
    return out.asnumpy()


def cmp_min(a, b):
    print(a)
    print(b)

    out = np.minimum(a, b)
    print(out)
    out_me = me_min(a, b)
    print(out_me)


def test_minimum_2_2():
    a = np.random.randn(2, 2, 1, 1).astype(np.float32)
    b = np.random.randn(2, 2, 1, 1).astype(np.float32)
    cmp_min(a, b)
