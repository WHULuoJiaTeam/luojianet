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
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.scatternd = P.ScatterNd()

    def construct(self, indices, update):
        return self.scatternd(indices, update, (3, 3))


arr_indices = np.array([[0, 1], [1, 1]]).astype(np.int32)
arr_update = np.array([3.2, 1.1]).astype(np.float32)


def test_net():
    scatternd = Net()
    print(arr_indices)
    print(arr_update)
    output = scatternd(Tensor(arr_indices), Tensor(arr_update))
    print(output.asnumpy())
