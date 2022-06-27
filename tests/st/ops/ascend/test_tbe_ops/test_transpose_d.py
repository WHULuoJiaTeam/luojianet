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
from luojianet_ms.ops import operations as P
from luojianet_ms.train.model import Model

context.set_context(device_target="Ascend")


class Net(nn.Module):
    def __init__(self, perm_in):
        super(Net, self).__init__()
        self.transpose = P.Transpose()
        self.perm = perm_in

    def forward(self, input_):
        x = self.transpose(input_, self.perm)
        return x


def ms_transpose(input_, perm_in):
    context.set_context(mode=context.GRAPH_MODE)
    input_me = Tensor(input_)
    net = Net(perm_in)
    net.set_train()
    model = Model(net)
    output = model.predict(input_me)
    print("-------------ms------------------")
    print(output.asnumpy().dtype)
    print(output.asnumpy())


def test_net():
    input_ = np.random.randn(8, 24, 1, 1).astype(np.float16)
    perm = (0, 2, 3, 1)
    ms_transpose(input_, perm)
