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

import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.composite as C
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.AddN()

    def construct(self, x, y):
        return self.add((x, y))


def test_net():
    x = np.random.randn(1, 3, 3, 4).astype(np.float32)
    y = np.random.randn(1, 3, 3, 4).astype(np.float32)
    add = Net()
    output = add(Tensor(x), Tensor(y))
    print(x)
    print(y)
    print(output.asnumpy())
    x = 1.0
    y = 2.0
    expect = 3.0
    add = Net()
    output = add(x, y)
    assert output == expect


def test_grad_addn_with_list():
    grad_op = C.GradOperation(get_all=True)
    class AddN(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add_n = P.AddN()

        def construct(self, a, b):
            return self.add_n([a, b])

    inp = Tensor(np.ones([128, 96]).astype(np.float32))
    grad_op(AddN())(inp, inp)
