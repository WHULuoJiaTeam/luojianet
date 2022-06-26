# Copyright 2021 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class InTopKNet(nn.Cell):
    def __init__(self, k):
        super(InTopKNet, self).__init__()
        self.in_top_k = P.InTopK(k)

    def construct(self, predictions, targets):
        return self.in_top_k(predictions, targets)


def in_top_k(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    predictions = Tensor(np.array([[4, 1, 2, 0, 0, 0, 0, 0, 0],
                                   [7, 9, 9, 0, 0, 0, 0, 0, 0],
                                   [3, 3, 3, 0, 0, 0, 0, 0, 0]]).astype(nptype))
    k = 165
    in_top_k_net = InTopKNet(k)
    targets = Tensor(np.array([0, 1, 0]).astype(np.int32))
    output = in_top_k_net(predictions, targets)
    expected_output = np.array([True, True, True])
    np.testing.assert_array_equal(output.asnumpy(), expected_output)

    k = -2
    in_top_k_net = InTopKNet(k)
    targets = Tensor(np.array([0, 1, 0]).astype(np.int32))
    output = in_top_k_net(predictions, targets)
    expected_output = np.array([False, False, False])
    np.testing.assert_array_equal(output.asnumpy(), expected_output)

    k = 1
    in_top_k_net = InTopKNet(k)
    targets = Tensor(np.array([0, 1, 0]).astype(np.int32))
    output = in_top_k_net(predictions, targets)
    expected_output = np.array([True, True, True])
    np.testing.assert_array_equal(output.asnumpy(), expected_output)

    k = 2
    in_top_k_net = InTopKNet(k)
    targets = Tensor(np.array([0, 1, 2]).astype(np.int32))
    output = in_top_k_net(predictions, targets)
    expected_output = np.array([True, True, True])
    np.testing.assert_array_equal(output.asnumpy(), expected_output)

    targets = Tensor(np.array([2, 2, 0]).astype(np.int32))
    output = in_top_k_net(predictions, targets)
    expected_output = np.array([True, True, True])
    np.testing.assert_array_equal(output.asnumpy(), expected_output)

    k = 3
    in_top_k_net = InTopKNet(k)
    targets = Tensor(np.array([2, 2, 2]).astype(np.int32))
    output = in_top_k_net(predictions, targets)
    expected_output = np.array([True, True, True])
    np.testing.assert_array_equal(output.asnumpy(), expected_output)

    targets = Tensor(np.array([1, 1, 0]).astype(np.int32))
    output = in_top_k_net(predictions, targets)
    expected_output = np.array([True, True, True])
    np.testing.assert_array_equal(output.asnumpy(), expected_output)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_in_top_k_float16():
    """
    Feature: Test InTopK op.
    Description: Test InTopK with float16 input.
    Expectation: The result match to the expect value.
    """
    in_top_k(np.float16)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_in_top_k_float32():
    """
    Feature: Test InTopK op.
    Description: Test InTopK with float input.
    Expectation: The result match to the expect value.
    """
    in_top_k(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_in_top_k_invalid_input():
    """
    Feature: Test InTopK op.
    Description: Test InTopK with invalid input.
    Expectation: Raise ValueError.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    # predictions must be 2d
    with pytest.raises(ValueError):
        in_top_k_net = InTopKNet(1)
        predictions = Tensor(np.zeros(4).astype(np.float32))
        targets = Tensor(np.zeros(4).astype(np.int32))
        _ = in_top_k_net(predictions, targets)

    # targets must be 1d
    with pytest.raises(ValueError):
        in_top_k_net = InTopKNet(1)
        predictions = Tensor(np.zeros(4).astype(np.float32))
        targets = Tensor(np.zeros(4).reshape(2, 2).astype(np.int32))
        _ = in_top_k_net(predictions, targets)

    # predictions.shape[1] must be equal to targets.shape[0]
    with pytest.raises(ValueError):
        in_top_k_net = InTopKNet(1)
        predictions = Tensor(np.zeros(4).reshape(2, 2).astype(np.float32))
        targets = Tensor(np.zeros(4).astype(np.int32))
        _ = in_top_k_net(predictions, targets)
