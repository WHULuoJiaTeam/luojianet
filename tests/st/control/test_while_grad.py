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
from luojianet_ms.ops import composite as C
import luojianet_ms.common.dtype as mstype
import luojianet_ms.nn as nn
import luojianet_ms.context as context
from luojianet_ms.common.tensor import Tensor

class Net(nn.Module):
    def call(self, x, y):
        while x < y:
            x = x * x + 1
        return x


class GradNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad_op = C.GradOperation(get_all=True)

    def call(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_grad():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    x = Tensor([2.0], dtype=mstype.float32)
    y = Tensor([2.0], dtype=mstype.float32)
    GradNet(Net())(x, y)
