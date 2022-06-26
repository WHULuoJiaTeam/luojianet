# Copyright 2020 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import ms_function
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = P.MatMul()

    @ms_function
    def construct(self, x1_, x2_):
        return self.matmul(x1_, x2_)


x1 = np.random.randn(1, 3).astype(np.float32)
x2 = np.random.randn(3, 4).astype(np.float32)


def test_net():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    matmul = Net()
    output = matmul(Tensor(x1), Tensor(x2))
    print(x1)
    print(x2)
    print(output.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    matmul = Net()
    output = matmul(Tensor(x1), Tensor(x2))
    print(x1)
    print(x2)
    print(output.asnumpy())
