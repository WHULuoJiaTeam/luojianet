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

import luojianet_ms.context as context
import luojianet_ms.nn as nn
from luojianet_ms import Tensor
from luojianet_ms.common.api import ms_function
from luojianet_ms.common.initializer import initializer
from luojianet_ms.common.parameter import Parameter
from luojianet_ms.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

        self.cat = P.Slice()
        self.x1 = Parameter(initializer(
            Tensor(np.array([[[1, -1, 1], [2, -2, 2]], [[3, -3, 3], [4, -4, 4]], [[5, -5, 5], [6, -6, 6]]]).astype(
                np.float32)), [3, 2, 3]), name='x1')

    @ms_function
    def forward(self):
        return self.cat(self.x1, (0, 1, 0), (2, 1, 3))


def test_slice():
    cat = Slice()
    output = cat()
    expect = [[[2., -2., 2.]],
              [[4., -4., 4.]]]
    print(output)
    assert (output.asnumpy() == expect).all()
