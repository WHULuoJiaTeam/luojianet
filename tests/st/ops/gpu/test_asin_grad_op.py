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
from luojianet_ms import Tensor
import luojianet_ms.ops.operations._grad_ops as P
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_asingrad_fp32():
    error = np.ones(4) * 1.0e-7
    x_np = np.array([0, -0.25, 0.5, 0.3]).astype(np.float32)
    dout_np = np.array([1, 1, 1, 1]).astype(np.float32)
    output_ms = P.AsinGrad()(Tensor(x_np), Tensor(dout_np))
    expect = np.array([1, 1.0327955, 1.1547005, 1.0482849])
    diff = output_ms.asnumpy() - expect
    assert np.all(diff < error)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_asingrad_fp16():
    error = np.ones(4) * 1.0e-3
    x_np = np.array([0, -0.25, 0.5, 0.3]).astype(np.float16)
    dout_np = np.array([1, 1, 1, 1]).astype(np.float16)
    output_ms = P.AsinGrad()(Tensor(x_np), Tensor(dout_np))
    expect = np.array([1, 1.033, 1.154, 1.048])
    diff = output_ms.asnumpy() - expect
    assert np.all(diff < error)
