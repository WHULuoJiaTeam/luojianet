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
from luojianet_ms.common import dtype as mstype
from luojianet_ms import nn
from luojianet_ms import Tensor
from luojianet_ms.ops import composite as C
from luojianet_ms import context

context.set_context(mode=context.GRAPH_MODE)


class ForwardNet(nn.Module):
    def __init__(self, max_cycles=10):
        super(ForwardNet, self).__init__()
        self.max_cycles = max_cycles
        self.zero = Tensor(np.array(0), mstype.int32)
        self.i = Tensor(np.array(0), mstype.int32)

    def forward(self, x, y):
        out = self.zero
        j = self.i
        while j < self.max_cycles:
            for _ in range(0, self.max_cycles):
                out = x * y + out
            j = j + 1
        i = self.i
        while i < self.max_cycles:
            out = x * y + out
            i = i + 1
        return out


class BackwardNet(nn.Module):
    def __init__(self, net):
        super(BackwardNet, self).__init__(auto_prefix=False)
        self.forward_net = net
        self.grad = C.GradOperation(get_all=True)

    def forward(self, *inputs):
        grads = self.grad(self.forward_net)(*inputs)
        return grads

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_forward():
    x = Tensor(np.array(1), mstype.int32)
    y = Tensor(np.array(3), mstype.int32)
    forward_net = ForwardNet(max_cycles=3)
    out = forward_net(x, y)
    expect_forward_res = Tensor([36], mstype.int32)
    assert out == expect_forward_res

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_backward():
    x = Tensor(np.array(1), mstype.int32)
    y = Tensor(np.array(3), mstype.int32)
    forward_net = ForwardNet(max_cycles=3)
    backward_net = BackwardNet(forward_net)
    grads = backward_net(x, y)
    expect_backward_res = (Tensor([36], mstype.int32), Tensor([12], mstype.int32))
    assert grads == expect_backward_res
