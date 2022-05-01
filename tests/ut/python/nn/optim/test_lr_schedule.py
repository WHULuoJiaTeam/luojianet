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
""" test_lr_schedule """
import numpy as np

from luojianet_ms import Parameter, ParameterTuple, Tensor
from luojianet_ms.nn import Module
from luojianet_ms.nn.optim import Optimizer
from luojianet_ms.ops.operations import BiasAdd, MatMul
import luojianet_ms.ops.composite as C


grad_by_list = C.GradOperation(get_by_list=True)


class Net(Module):
    """ Net definition """

    def __init__(self):
        super(Net, self).__init__()
        self.weight = Parameter(Tensor(np.ones([64, 10])), name="weight")
        self.bias = Parameter(Tensor(np.ones([10])), name="bias")
        self.matmul = MatMul()
        self.biasAdd = BiasAdd()

    def call(self, x):
        x = self.biasAdd(self.matmul(x, self.weight), self.bias)
        return x


class _TrainOneStepCell(Module):
    """ _TrainOneStepCell definition """

    def __init__(self, network, optimizer):
        """
        Append an optimizer to the training network after that the call
        function can be called to create the backward graph.
        Arguments:
            network: The training network.
                Note that loss function should have been added.
            optimizer: optimizer for updating the weights
        """
        super(_TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(network.get_parameters())

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an optimizer'.format(
                type(optimizer).__name__))

        self.has_lr_schedule = False
        self.optimizer = optimizer

    def call(self, data, label, *args):
        weights = self.weights
        grads = grad_by_list(self.network, weights)(data, label)
        if self.lr_schedule:
            self.schedule.update_lr(*args)
        return self.optimizer(grads)
