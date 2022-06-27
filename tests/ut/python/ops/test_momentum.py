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
""" test_momentum """
import functools
import numpy as np

import luojianet_ms.nn as nn
import luojianet_ms.context as context
from luojianet_ms import Parameter, ParameterTuple, Tensor
from luojianet_ms.ops import composite as C
from luojianet_ms.ops import functional as F
from luojianet_ms.ops import operations as P
from ..ut_filter import non_graph_engine
from ....luojianet_ms_test_framework.luojianet_ms_test import luojianet_ms_test
from ....luojianet_ms_test_framework.pipeline.forward.compile_forward \
    import pipeline_for_compile_forward_ge_graph_for_case_by_case_config

# pylint: disable=W0613
# W0613: unused-argument


run_opt = C.MultitypeFuncGraph("run_opt")


grad_by_list = C.GradOperation(get_by_list=True)


@run_opt.register("Function", "Tensor", "Tensor", "Tensor",
                  "Tensor", "Tensor",
                  "Tensor")
def tensor_run_opt(opt, iters, learning_rate, momentum,
                   gradient, variable, moment):
    """ tensor_run_opt """
    success = True
    new_weight = opt(variable, moment, learning_rate, gradient, momentum)
    success = F.depend(success, F.assign(variable, new_weight))
    return success


class OptimizerByMomentum(nn.Module):
    """ OptimizerByMomentum definition """

    def __init__(self, weights):
        super(OptimizerByMomentum, self).__init__()
        self.learning_rate = Parameter(0.1, name="learning_rate")
        self.momentum = Parameter(0.05, name="momentum")
        self.iter = Parameter(0, name="iter")

        self.weights = weights
        self.moments = weights.clone(prefix="moments", init='zeros')

        self.hyper_map = C.HyperMap()
        self.opt = P.ApplyMomentum()

    def forward(self, grads):
        success = True
        weights = self.weights
        moments = self.moments
        success = self.hyper_map(F.partial(run_opt, self.opt, self.iter,
                                           self.learning_rate, self.momentum),
                                 grads, weights, moments)
        return success


class TrainStepWrap(nn.Module):
    """ TrainStepWrap definition """

    def __init__(self, network):
        super(TrainStepWrap, self).__init__()
        self.network = network
        self.weights = ParameterTuple(network.get_parameters())
        self.optimizer = OptimizerByMomentum(self.weights)
        self.hyper_map = C.HyperMap()

    def forward(self, x, label):
        weights = self.weights
        grads = grad_by_list(self.network, weights)(x, label)
        return self.optimizer(grads)


class NetWithLossClass(nn.Module):
    """ NetWithLossClass definition """

    def __init__(self, network):
        super(NetWithLossClass, self).__init__(auto_prefix=False)
        self.loss = nn.SoftmaxCrossEntropyWithLogits()
        self.network = network

    def forward(self, x, label):
        predict = self.network(x)
        return self.loss(predict, label)


class Net(nn.Module):
    """ Net definition """

    def __init__(self):
        super(Net, self).__init__()
        self.weight = Parameter(Tensor(np.ones([64, 10]).astype(np.float32)), name="weight")
        self.bias = Parameter(Tensor(np.ones([10]).astype(np.float32)), name="bias")
        self.matmul = P.MatMul()
        self.biasAdd = P.BiasAdd()

    def forward(self, x):
        return self.biasAdd(self.matmul(x, self.weight), self.bias)


test_case_ops = [
    ('Momentum', {
        'block': TrainStepWrap(NetWithLossClass(Net())),
        'desc_inputs': [Tensor(np.ones([1, 64]).astype(np.float32)),
                        Tensor(np.zeros([1, 10]).astype(np.float32))]}),
]

test_case_lists = [test_case_ops]
test_exec_case = functools.reduce(lambda x, y: x + y, test_case_lists)
# use -k to select certain testcast
# pytest tests/python/ops/test_ops.py::test_backward -k LayerNorm



@non_graph_engine
@luojianet_ms_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config)
def test_exec():
    context.set_context(mode=context.GRAPH_MODE)
    return test_exec_case
