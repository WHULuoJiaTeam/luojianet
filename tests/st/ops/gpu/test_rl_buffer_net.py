# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore.common.api import ms_function
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
import mindspore as ms


def create_tensor(capcity, shapes, dtypes):
    buffer = []
    for i in range(len(shapes)):
        buffer.append(Tensor(np.zeros(((capcity,)+shapes[i])), dtypes[i]))
    return buffer


class RLBuffer(nn.Cell):
    def __init__(self, batch_size, capcity, shapes, types):
        super(RLBuffer, self).__init__()
        self.buffer = create_tensor(capcity, shapes, types)
        self._capacity = capcity
        self.count = Parameter(Tensor(0, ms.int32), name="count")
        self.head = Parameter(Tensor(0, ms.int32), name="head")
        self.buffer_append = P.BufferAppend(self._capacity, shapes, types)
        self.buffer_get = P.BufferGetItem(self._capacity, shapes, types)
        self.buffer_sample = P.BufferSample(
            self._capacity, batch_size, shapes, types)
        self.randperm = P.Randperm(max_length=capcity, pad=-1)
        self.reshape = P.Reshape()

    @ms_function
    def append(self, exps):
        return self.buffer_append(self.buffer, exps, self.count, self.head)

    @ms_function
    def get(self, index):
        return self.buffer_get(self.buffer, self.count, self.head, index)

    @ms_function
    def sample(self):
        return self.buffer_sample(self.buffer, self.count, self.head)


s = Tensor(np.array([2, 2, 2, 2]), ms.float32)
a = Tensor(np.array([0, 1]), ms.int32)
r = Tensor(np.array([1]), ms.float32)
s_ = Tensor(np.array([3, 3, 3, 3]), ms.float32)
exp = [s, a, r, s_]
exp1 = [s_, a, r, s]


@ pytest.mark.level0
@ pytest.mark.platform_x86_gpu_training
@ pytest.mark.env_onecard
def test_Buffer():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    buffer = RLBuffer(batch_size=32, capcity=100, shapes=[(4,), (2,), (1,), (4,)], types=[
        ms.float32, ms.int32, ms.float32, ms.float32])
    print("init buffer:\n", buffer.buffer)
    for _ in range(0, 110):
        buffer.append(exp)
    buffer.append(exp1)
    print("buffer append:\n", buffer.buffer)
    b = buffer.get(-1)
    print("buffer get:\n", b)
    bs = buffer.sample()
    print("buffer sample:\n", bs)
