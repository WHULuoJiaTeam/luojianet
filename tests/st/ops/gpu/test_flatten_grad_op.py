# Copyright 2019 Huawei Technologies Co., Ltd
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
from mindspore.ops.operations import _grad_ops as G


class NetFlattenGrad(nn.Cell):
    def __init__(self):
        super(NetFlattenGrad, self).__init__()
        self.flattengrad = G.FlattenGrad()
        self.type = (2, 3)

    def construct(self, x):
        return self.flattengrad(x, self.type)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_flatten_grad():
    x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float32))
    expect = np.array([[-0.1, 0.3, 3.6],
                       [0.4, 0.5, -3.2]]).astype(np.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    flattengrad = NetFlattenGrad()
    output = flattengrad(x)
    assert (output.asnumpy() == expect).all()
