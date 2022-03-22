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

from luojianet_ms import context
from luojianet_ms.common.tensor import Tensor
from luojianet_ms.nn import Cell
from luojianet_ms.ops import operations as P
from luojianet_ms.ops.composite import GradOperation

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Grad(Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_, output_grad):
        gout = self.grad(self.network)(input_, output_grad)
        return gout


class Net(Cell):
    def __init__(self, begin, end, stride):
        super(Net, self).__init__()
        self.stridedslice = P.StridedSlice()
        self.begin = begin
        self.end = end
        self.stride = stride

    def construct(self, input_):
        x = self.stridedslice(input_, self.begin, self.end, self.stride)
        return x


def me_stridedslice(input_, begin, end, stride, gradients):
    input_me = Tensor(input_)
    out_grad_me = Tensor(gradients)
    net_me = Grad(Net(begin, end, stride))
    net_me.set_train()
    out_grad = net_me(input_me, out_grad_me)
    print(out_grad.asnumpy())


def test_grad_stridedslice_1d():
    input_ = np.random.randn(2).astype(np.float32)
    begin = (0,)
    end = (2,)
    stride = (1,)
    gradients = np.random.randn(2).astype(np.float32)
    me_stridedslice(input_, begin, end, stride, gradients)
