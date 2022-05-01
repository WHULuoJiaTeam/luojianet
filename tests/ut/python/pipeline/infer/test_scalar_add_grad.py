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
""" test_scalar_add_grad """
import numpy as np

from luojianet_ms import context
from luojianet_ms.common.tensor import Tensor
from luojianet_ms.nn import Module
from luojianet_ms.nn import ReLU
from luojianet_ms.ops import composite as C
from luojianet_ms.ops.operations import Add

context.set_context(mode=context.GRAPH_MODE)
grad = C.GradOperation(get_all=True, sens_param=True)


class TensorAddNetMe(Module):
    """ TensorAddNetMe definition """

    def __init__(self):
        super(TensorAddNetMe, self).__init__()
        self.relu = ReLU()
        self.add = Add()

    def call(self, inputA, inputB):
        inputA = self.relu(inputA)
        inputB = self.relu(inputB)
        x = self.add(inputA, inputB)
        x = self.relu(x)
        return x


class GradWrap2(Module):
    """ GradWrap2 definition """

    def __init__(self, network):
        super(GradWrap2, self).__init__()
        self.network = network

    def call(self, inputA, inputB, sens):
        gout = grad(self.network)(inputA, inputB, sens)
        return gout


def gen_forwarddata(inputA, inputB):
    """ gen_forwarddata """
    net_me = TensorAddNetMe()
    net_me.set_train()
    output = net_me(Tensor(inputA), Tensor(inputB))
    print(output)


def gen_backwarddata(inputA, inputB, inputGrad):
    """ gen_backwarddata """
    net_me = GradWrap2(TensorAddNetMe())
    net_me.set_train()
    output = net_me(Tensor(inputA), Tensor(inputB), Tensor(inputGrad))
    print(output)


def test_scalar_tennsor_add():
    """ test_scalar_tennsor_add """
    inputa = np.array(32).astype(np.float32)
    inputb = np.random.randn(1280, 768).astype(np.float32)
    gen_forwarddata(inputa, inputb)


def test_scalar_tennsor_gradadd():
    """ test_scalar_tennsor_gradadd """
    inputa = np.array(32).astype(np.float32)
    inputb = np.random.randn(1280, 768).astype(np.float32)
    inputgrad = np.random.randn(1280, 768).astype(np.float32)
    gen_backwarddata(inputa, inputb, inputgrad)
