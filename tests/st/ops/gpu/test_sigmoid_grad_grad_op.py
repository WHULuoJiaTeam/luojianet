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
import pytest

import luojianet_ms.context as context
import luojianet_ms.nn as nn
from luojianet_ms import Tensor
from luojianet_ms.ops.operations import _grad_ops as G
from luojianet_ms.ops.composite import GradOperation


class NetSigmoidGrad(nn.Module):
    def __init__(self):
        super(NetSigmoidGrad, self).__init__()
        self.sigmoid_grad = G.SigmoidGrad()

    def call(self, y, dy):
        return self.sigmoid_grad(y, dy)


class Grad(nn.Module):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    def call(self, y, y_grad, dout):
        return self.grad(self.network)(y, y_grad, dout)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sigmoid_grad_grad():
    y = Tensor(np.array([[[[-1, 1, 2],
                           [1, -1, 1],
                           [2, 1, -1]]]]).astype(np.float32))
    y_grad = Tensor(np.array([[[[-11, 2, 4],
                                [-1, 1, -1],
                                [-4, 4, -4]]]]).astype(np.float32))
    dout = Tensor(np.array([[[[-11, 2, 4],
                              [-1, 1, -1],
                              [-4, 4, -4]]]]).astype(np.float32))

    expect_ddy = np.array([[[[363., -4., -48.],
                             [-1., 3., -1.],
                             [-48., -16., 48.]]]]).astype(np.float32)

    expect_d2x = np.array([[[[22., 0., -8.],
                             [-0., -2., -0.],
                             [8., 0., 8.]]]]).astype(np.float32)

    error = np.ones(shape=[1, 1, 3, 3]) * 1.0e-6

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    sigmoid_grad_grad = Grad(NetSigmoidGrad())
    ddy, d2x = sigmoid_grad_grad(y, y_grad, dout)
    diff0 = ddy.asnumpy() - expect_ddy
    diff1 = d2x.asnumpy() - expect_d2x
    assert np.all(abs(diff0) < error)
    assert np.all(abs(diff1) < error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    sigmoid_grad_grad = Grad(NetSigmoidGrad())
    ddy, d2x = sigmoid_grad_grad(y, y_grad, dout)
    diff0 = ddy.asnumpy() - expect_ddy
    diff1 = d2x.asnumpy() - expect_d2x
    assert np.all(abs(diff0) < error)
    assert np.all(abs(diff1) < error)
