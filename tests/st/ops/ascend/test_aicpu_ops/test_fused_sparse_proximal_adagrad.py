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
import luojianet_ms.nn as nn
import luojianet_ms.context as context
import luojianet_ms.common.dtype as mstype
from luojianet_ms import Tensor
from luojianet_ms.ops import operations as P
from luojianet_ms.common.parameter import Parameter

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fused_sparse_proximal_adagrad = P.FusedSparseProximalAdagrad()
        self.var = Parameter(Tensor(np.ones([3, 3]).astype(np.float32)), name="var")
        self.accum = Parameter(Tensor(np.ones([3, 3]).astype(np.float32)), name="accum")
        self.lr = 0.01
        self.l1 = 0.0
        self.l2 = 0.0

    def call(self, grad, indices):
        return self.fused_sparse_proximal_adagrad(self.var, self.accum, self.lr, self.l1, self.l2,
                                                  grad, indices)

def test_net():
    gradient = Tensor(np.array([-3, 2, 3, 0, 0, 0, -4, -1, -2])
                      .reshape([3, 3]).astype(np.float32))
    indices = Tensor(np.ones([3]), mstype.int32)
    net = Net()
    output = net(gradient, indices)
    print(output)
    print(net.var.data)
    print(net.accum.data)
