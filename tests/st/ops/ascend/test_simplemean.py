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
from mindspore import Tensor
from mindspore.common.api import ms_function
from mindspore.ops import operations as P

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.simplemean = P.ReduceMean(keep_dims=True)

    @ms_function
    def construct(self, x):
        return self.simplemean(x, (-2, -1))


def test_net():
    x = np.random.randn(32, 2048, 7, 7).astype(np.float32)
    simplemean = Net()
    output = simplemean(Tensor(x))
    print(output.asnumpy())
