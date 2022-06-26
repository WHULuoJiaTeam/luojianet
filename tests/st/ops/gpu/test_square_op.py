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
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops import operations as P


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_square_normal():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_np = np.random.rand(2, 3, 4, 4).astype(np.float32)
    output_ms = P.Square()(Tensor(x_np))
    output_np = np.square(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)
    x_np = np.random.rand(2, 3, 1, 5, 4, 4).astype(np.float32)
    output_ms = P.Square()(Tensor(x_np))
    output_np = np.square(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)
    x_np = np.random.rand(2,).astype(np.float32)
    output_ms = P.Square()(Tensor(x_np))
    output_np = np.square(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)


# Dynamic Shape Testing
class SqaureNetDynamic(nn.Cell):
    def __init__(self):
        super(SqaureNetDynamic, self).__init__()
        self.square = P.Square()
        self.gpu_convert_to_dynamic_shape = inner.GpuConvertToDynamicShape()

    def construct(self, x):
        x_dyn = self.gpu_convert_to_dynamic_shape(x)
        return self.square(x_dyn)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_square_dynamic():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = SqaureNetDynamic()
    x_np = np.random.rand(1, 3, 4, 4, 1).astype(np.float32)
    output_ms = net(Tensor(x_np))
    output_np = np.square(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)
    x_np = np.random.rand(2, 3, 4, 4, 8, 9).astype(np.float16)
    output_ms = net(Tensor(x_np))
    output_np = np.square(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)
    x_np = np.random.rand(1).astype(np.float32)
    output_ms = net(Tensor(x_np))
    output_np = np.square(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)
