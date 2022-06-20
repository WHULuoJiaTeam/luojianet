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
import pytest
from luojianet_ms import context
from luojianet_ms import Tensor, nn
from luojianet_ms.ops import composite as C
from luojianet_ms.common import dtype as mstype
from luojianet_ms.common.parameter import Parameter

grad_all = C.GradOperation(get_all=True)


@pytest.mark.skip(reason="not supported for in while")
def test_if_after_for_in_while():
    class IfAfterForInWhileNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.param_a = Parameter(Tensor(5, mstype.int32), name='a')
            self.param_b = Parameter(Tensor(2, mstype.int32), name='b')

        def call(self, x):
            out = x + self.param_a
            while self.param_a > self.param_b:
                self.param_b += 1
                for _ in range(4):
                    self.param_a += 3
            self.param_a -= 40
            if x > self.param_a:
                out += self.param_a * 10
            return out

    class GradNet(nn.Module):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def call(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor(2, mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    if_after_for_in_while_net = IfAfterForInWhileNet()
    net = GradNet(if_after_for_in_while_net)

    forward_net = IfAfterForInWhileNet()
    graph_forward_res = forward_net(x)
    graph_backward_res = net(x)

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    if_after_for_in_while_net = IfAfterForInWhileNet()
    net = GradNet(if_after_for_in_while_net)

    forward_net = IfAfterForInWhileNet()
    pynative_forward_res = forward_net(x)
    pynative_backward_res = net(x)

    assert graph_forward_res == pynative_forward_res
    assert graph_backward_res == pynative_backward_res
