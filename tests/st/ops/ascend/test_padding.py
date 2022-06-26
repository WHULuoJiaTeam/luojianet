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
import luojianet_ms.common.dtype as mstype
from luojianet_ms import Tensor
from luojianet_ms.ops import operations as P

context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, pad_dim_size):
        super(Net, self).__init__()
        self.padding = P.Padding(pad_dim_size)

    def construct(self, x):
        return self.padding(x)


def test_padding():
    x = Tensor(np.array([[8], [10]]), mstype.int32)
    padding = Net(4)
    out = padding(x)
    assert(out.asnumpy() == [[8, 0, 0, 0], [10, 0, 0, 0]]).all()
