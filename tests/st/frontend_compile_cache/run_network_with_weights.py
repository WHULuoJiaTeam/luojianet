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
import sys
import numpy as np

import luojianet_ms.context as context
import luojianet_ms.nn as nn
from luojianet_ms import Tensor, Parameter
from luojianet_ms import dtype as mstype
from luojianet_ms.ops import operations as P


class NetWithWeights(nn.Module):
    def __init__(self):
        super(NetWithWeights, self).__init__()
        self.matmul = P.MatMul()
        self.a = Parameter(Tensor(np.array([2.0], np.float32)), name='a')
        self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

    def call(self, x, y):
        x = x * self.z
        y = y * self.a
        out = self.matmul(x, y)
        return out


def run_simple_net():
    x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.11, 3.3, 1.1], [1.1, 0.2, 1.4], [1.1, 2.2, 0.3]], dtype=mstype.float32)
    net = NetWithWeights()
    output = net(x, y)
    print("{", output, "}")
    print("{", output.asnumpy().shape, "}")


if __name__ == "__main__":
    context.set_context(enable_compile_cache=True, compile_cache_path=sys.argv[1])
    run_simple_net()
    context.set_context(enable_compile_cache=False)
