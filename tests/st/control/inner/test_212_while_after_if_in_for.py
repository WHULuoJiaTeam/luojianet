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
from luojianet_ms.common.parameter import Parameter

context.set_context(mode=context.GRAPH_MODE)


class ForwardNet(nn.Module):
    def __init__(self, max_cycles=10):
        super(ForwardNet, self).__init__()
        self.max_cycles = max_cycles
        self.i = Tensor(np.array(0), mstype.int32)
        self.zero = Tensor(np.array(0), mstype.int32)
        self.weight = Parameter(Tensor(np.array(0), mstype.int32))

    def call(self, x, y):
        i = self.i
        out = self.zero
        for _ in range(0, self.max_cycles):
            if out <= 20:
                self.weight = out
                out = x * y + out
        while i < self.max_cycles:
            out = out + 10
            i = i + 1
            self.weight = self.weight - i
        out1 = self.weight + 1
        return out, out1


class BackwardNet(nn.Module):
    def __init__(self, net):
        super(BackwardNet, self).__init__(auto_prefix=False)
        self.forward_net = net
        self.grad = C.GradOperation(get_all=True)

    def call(self, *inputs):
        grads = self.grad(self.forward_net)(*inputs)
        return grads


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_forward():
    x = Tensor(np.array(3), mstype.int32)
    y = Tensor(np.array(5), mstype.int32)
    # Graph Mode
    context.set_context(mode=context.GRAPH_MODE)
    graph_forward_net = ForwardNet(max_cycles=3)
    graph_mode_out = graph_forward_net(x, y)

    assert graph_mode_out == (Tensor(np.array(60), mstype.int32), Tensor(np.array(10), mstype.int32))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_backward():
    x = Tensor(np.array(3), mstype.int32)
    y = Tensor(np.array(5), mstype.int32)
    # Graph Mode
    context.set_context(mode=context.GRAPH_MODE)
    graph_forward_net = ForwardNet(max_cycles=3)
    graph_backward_net = BackwardNet(graph_forward_net)
    graph_mode_grads = graph_backward_net(x, y)

    assert graph_mode_grads == (Tensor(np.array(10), mstype.int32), Tensor(np.array(6), mstype.int32))


class ForwardNetNoAssign(nn.Module):
    def __init__(self, max_cycles=10):
        super(ForwardNetNoAssign, self).__init__()
        self.max_cycles = max_cycles
        self.i = Tensor(np.array(0), mstype.int32)
        self.zero = Tensor(np.array(0), mstype.int32)
        self.weight = Parameter(Tensor(np.array(0), mstype.int32))

    def call(self, x, y):
        i = self.i
        out = self.zero
        for _ in range(0, self.max_cycles):
            if out <= 20:
                out = x * y + out
        while i < self.max_cycles:
            out = out + 10
            i = i + 1
        return out


class BackwardNetNoAssign(nn.Module):
    def __init__(self, net):
        super(BackwardNetNoAssign, self).__init__(auto_prefix=False)
        self.forward_net = net
        self.grad = C.GradOperation(get_all=True)

    def call(self, *inputs):
        grads = self.grad(self.forward_net)(*inputs)
        return grads


def test_backward_no_assign():
    x = Tensor(np.array(3), mstype.int32)
    y = Tensor(np.array(5), mstype.int32)
    # Graph Mode
    context.set_context(mode=context.GRAPH_MODE)
    graph_forward_net = ForwardNetNoAssign(max_cycles=3)
    graph_backward_net = BackwardNetNoAssign(graph_forward_net)
    graph_mode_grads = graph_backward_net(x, y)

    assert graph_mode_grads == (Tensor(np.array(10), mstype.int32), Tensor(np.array(6), mstype.int32))
