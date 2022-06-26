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

#!/usr/bin/env python3

import numpy as np
import pytest

from mindspore import context, Tensor
from mindspore.common import dtype as mstype
from mindspore import ms_function
from mindspore import ops

context.set_context(mode=context.PYNATIVE_MODE)
input_x = Tensor(np.ones([1, 1, 120, 640]), dtype=mstype.float32)
input_y = Tensor(np.full((1, 1, 120, 640), 4), dtype=mstype.float32)
ret_output_2 = Tensor(np.full((1, 1, 120, 640), 3.125), dtype=mstype.float32)


@pytest.mark.level1
@pytest.mark.timeout(60)
@pytest.mark.env_Ascend_1p
@pytest.mark.env_Gpu_1p
@pytest.mark.env_CPU
@pytest.mark.Function
def test_ms_function_nested_local():
    @ms_function
    def function1(x, y):
        x = x ** y
        x /= y
        x += y
        x -= 1
        x %= 2
        return x

    @ms_function
    def function11(x, y):
        r = function1(x, y)
        out = r + r
        return out

    @ms_function
    def function2(x, y):
        r1 = function1(x, y)
        r2 = function11(x, y)
        z = r1 * r2
        return z

    output2 = function2(input_x, input_y)
    assert np.allclose(output2.asnumpy(), ret_output_2.asnumpy(), 0.0001, 0.0001)


@ms_function
def function1_g(x, y):
    x = x ** y
    x /= y
    x += y
    x -= 1
    x %= 2
    return x

@ms_function
def function11_g(x, y):
    r = function1_g(x, y)
    out = r + r
    return out

@pytest.mark.level1
@pytest.mark.timeout(60)
@pytest.mark.env_Ascend_1p
@pytest.mark.env_Gpu_1p
@pytest.mark.env_CPU
@pytest.mark.Function
def test_ms_function_nested_global():
    @ms_function
    def function2_g(x, y):
        r1 = function1_g(x, y)
        r2 = function11_g(x, y)
        z = r1 * r2
        return z

    output2 = function2_g(input_x, input_y)
    assert np.allclose(output2.asnumpy(), ret_output_2.asnumpy(), 0.0001, 0.0001)


@pytest.mark.level1
@pytest.mark.timeout(60)
@pytest.mark.env_Ascend_1p
@pytest.mark.env_Gpu_1p
@pytest.mark.env_CPU
@pytest.mark.Function
def test_ms_function_nested_grad():
    """
    Feature: Nested call of ms_function
    Description: test nested call of ms_function
    Expectation: First derivative 75, Second derivative 30
    """
    x = Tensor([5], dtype=mstype.float32)
    exp1 = Tensor([75], dtype=mstype.float32)
    exp2 = Tensor([30], dtype=mstype.float32)
    def f(x):
        return x**3

    # 一阶：3*x^2 = 75
    out = ms_function(ops.grad(f))(x)
    assert np.allclose(out[0].asnumpy(), exp1[0].asnumpy(), 0.0001, 0.0001)
    out = ms_function(ms_function(ops.grad(f)))(x)
    assert np.allclose(out[0].asnumpy(), exp1[0].asnumpy(), 0.0001, 0.0001)

    # 二阶：6*x = 30
    out = ops.grad(ops.grad(f))(x)
    assert np.allclose(out[0].asnumpy(), exp2[0].asnumpy(), 0.0001, 0.0001)
    out = ms_function(ops.grad(ops.grad(f)))(x)
    assert np.allclose(out[0].asnumpy(), exp2[0].asnumpy(), 0.0001, 0.0001)
    out = ms_function(ms_function(ops.grad(ops.grad(f))))(x)
    assert np.allclose(out[0].asnumpy(), exp2[0].asnumpy(), 0.0001, 0.0001)
