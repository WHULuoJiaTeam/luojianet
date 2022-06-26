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
import mindspore.common.dtype as mstype
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

beta1_power = 0.9
beta2_power = 0.999
lr = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.fused_sparse_lazy_adam = P.FusedSparseLazyAdam()
        self.var = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="var")
        self.m = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="m")
        self.v = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="v")

    def construct(self, grad, indices):
        return self.fused_sparse_lazy_adam(self.var, self.m, self.v, beta1_power, beta2_power,
                                           lr, beta1, beta2, epsilon, grad, indices)

def test_net():
    gradient = Tensor(np.array([0.22948648, 0.14569908, 0.92861906, 0.66870148])
                      .reshape([2, 1, 2]).astype(np.float32))
    indices = Tensor([0, 1], mstype.int32)
    net = Net()
    output = net(gradient, indices)
    print(output)
    print(net.var.data)
    print(net.m.data)
    print(net.v.data)
