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


class ArgMax(nn.Cell):
    def __init__(self, axis):
        super(ArgMax, self).__init__()
        self.arg_max = P.Argmax(axis=axis)

    def construct(self, x):
        return self.arg_max(x)


def get_output(x, axis, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = ArgMax(axis)
    output = net(x)
    return output


def test_argmax():
    x0 = Tensor(np.random.normal(0, 1, [2, 3, 4, 4]).astype(np.float32))
    axis0 = 3
    expect = get_output(x0, axis0, False)
    output = get_output(x0, axis0, True)
    assert np.allclose(expect.asnumpy(), output.asnumpy(), 0.0001, 0.0001)

    x1 = Tensor(np.random.normal(0, 1, [2, 3, 1, 4]).astype(np.float32))
    axis1 = 2
    expect = get_output(x1, axis1, False)
    output = get_output(x1, axis1, True)
    assert np.allclose(expect.asnumpy(), output.asnumpy(), 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_argmax_gpu():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_argmax()
