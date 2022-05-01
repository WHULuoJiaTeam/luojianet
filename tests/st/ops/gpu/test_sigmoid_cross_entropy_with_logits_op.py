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
from luojianet_ms.ops import operations as P


class NetSigmoidCrossEntropyWithLogits(nn.Module):
    def __init__(self):
        super(NetSigmoidCrossEntropyWithLogits, self).__init__()
        self.loss = P.SigmoidCrossEntropyWithLogits()

    def call(self, logits, labels):
        return self.loss(logits, labels)


def sigmoid_cross_entropy_with_logits(nptype):
    logits = Tensor(np.array([[1, 1, 2],
                              [1, 2, 1],
                              [2, 1, 1]]).astype(nptype))
    labels = Tensor(np.array([[0, 0, 1],
                              [0, 1, 0],
                              [1, 0, 0]]).astype(nptype))
    expect_loss = np.array([[1.313262, 1.313262, 0.126928],
                            [1.313262, 0.126928, 1.313262],
                            [0.126928, 1.313262, 1.313262]]).astype(nptype)

    error = np.ones(shape=[3, 3]) * 1.0e-6

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetSigmoidCrossEntropyWithLogits()
    output = net(logits, labels)
    diff = output.asnumpy() - expect_loss
    assert np.all(abs(diff) < error)

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    net = NetSigmoidCrossEntropyWithLogits()
    output = net(logits, labels)
    diff = output.asnumpy() - expect_loss
    assert np.all(abs(diff) < error)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sigmoid_cross_entropy_with_logits_float32():
    sigmoid_cross_entropy_with_logits(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sigmoid_cross_entropy_with_logits_float64():
    sigmoid_cross_entropy_with_logits(np.float64)
