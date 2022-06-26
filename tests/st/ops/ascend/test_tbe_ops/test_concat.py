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
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()

        self.cat = P.Concat(axis=1)
        self.x1 = Parameter(initializer(
            Tensor(np.arange(2 * 2).reshape(2, 2).astype(np.float32)), [2, 2]), name='x1')
        self.x2 = Parameter(initializer(
            Tensor(np.arange(2 * 3).reshape(2, 3).astype(np.float32)), [2, 3]), name='x2')

    @ms_function
    def construct(self):
        return self.cat((self.x1, self.x2))


def test_net():
    cat = Net()
    output = cat()
    expect = [[0., 1., 0., 1., 2.],
              [2., 3., 3., 4., 5.]]
    print(np.arange(2 * 2).reshape(2, 2))
    print(np.arange(2 * 3).reshape(2, 3))
    print(output)
    assert (output.asnumpy() == expect).all()
