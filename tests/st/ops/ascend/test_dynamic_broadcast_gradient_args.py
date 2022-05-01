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

import luojianet_ms.nn as nn
import luojianet_ms.context as context

from luojianet_ms.ops.operations import _inner_ops

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.args = _inner_ops.DynamicBroadcastGradientArgs()

    def call(self, s0, s1):
        return self.args(s0, s1)


def test_net():
    shape0 = (4, 2, 1)
    shape1 = (2, 7)
    net = Net()
    r0, r1 = net(shape0, shape1)
    print(r0, r1)
    r0_expected = [2]
    r1_expected = [0]

    assert np.array_equal(r0_expected, r0.asnumpy())
    assert np.array_equal(r1_expected, r1.asnumpy())
