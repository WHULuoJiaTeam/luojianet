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
import luojianet_ms.context as context
import luojianet_ms.nn as nn
from luojianet_ms import Tensor, Parameter
import luojianet_ms.common.dtype as mstype
from luojianet_ms.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.unique = P.Unique()
        self.dynamic_assign = P.DynamicAssign()
        self.param = Parameter(
            Tensor(np.zeros((5,), np.int32)), name="assign_x")

    def call(self, y):
        y, _ = self.unique(y)
        return self.dynamic_assign(self.param, y)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_assign():
    y = Tensor(np.array([2, 2, 3, 3, 4]), mstype.int32)
    dynamic_assign = Net()
    _ = dynamic_assign(y)
    expect1 = np.array([2, 3, 4])
    param_np = dynamic_assign.param.data.asnumpy()
    assert (param_np == expect1).all()
