# Copyright 2019 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P


class AssignAdd(nn.Cell):
    def __init__(self, value):
        super(AssignAdd, self).__init__()
        self.var = Parameter(value, name="var")
        self.add = P.AssignAdd()

    def construct(self, y):
        res = self.add(self.var, y)
        return res


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_assign_add():
    expect1 = np.array([[[[0, 2, 4.],
                          [6, 8, 10.],
                          [12, 14, 16.]],
                         [[18, 20, 22.],
                          [24, 26, 28.],
                          [30, 32, 34.]],
                         [[36, 38, 40.],
                          [42, 44, 46.],
                          [48, 50, 52.]]]])
    expect2 = np.array([[[[0, 3, 6],
                          [9, 12, 15],
                          [18, 21, 24]],
                         [[27, 30, 33],
                          [36, 39, 42],
                          [45, 48, 51]],
                         [[54, 57, 60],
                          [63, 66, 69],
                          [72, 75, 78]]]])
    x1 = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float32))
    y1 = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float32))

    x2 = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float32))
    y2 = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float32))

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    add = AssignAdd(x1)
    output1 = add(y1)
    assert (output1.asnumpy() == expect1).all()
    add = AssignAdd(output1)
    output2 = add(y1)
    assert (output2.asnumpy() == expect2).all()

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    add = AssignAdd(x2)
    output1 = add(y2)
    assert (output1.asnumpy() == expect1).all()
    add = AssignAdd(output1)
    output2 = add(y2)
    assert (output2.asnumpy() == expect2).all()
