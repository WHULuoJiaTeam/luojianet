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
""" test graph fallback """
import pytest
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, ms_function, context
import mindspore.common.dtype as mstype

context.set_context(mode=context.GRAPH_MODE)


class ControlNet(nn.Cell):
    def inner_function_1(self, a, b):
        return a + b

    def inner_function_2(self, a, b):
        return a - b

    def construct(self, x):
        a = Tensor(np.array(4), mstype.int32)
        b = Tensor(np.array(5), mstype.int32)
        if a + b > x:
            return self.inner_function_1(a, b)
        return self.inner_function_2(a, b)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_control_sink_tensor():
    """
    Feature: Fallback feature: support define Tensor in Class construct.
    Description: Fallback feature: support define Tensor in Class construct.
    Expectation: Fallback feature: support define Tensor in Class construct.
    """
    x = Tensor(np.array(1), mstype.int32)
    net = ControlNet()
    output = net(x)
    output_expect = Tensor(9, mstype.int32)
    assert output == output_expect


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_tensor_list():
    """
    Feature: Fallback feature
    Description: support Basic method of Tensor list.
    Expectation: No exception.
    """
    @ms_function
    def np_tensor_list():
        a = Tensor(np.array(4), mstype.int32)
        b = Tensor(np.array(5), mstype.int32)
        c = Tensor(np.array(6), mstype.int32)
        tensor_list = [a, b]
        for tensor in tensor_list:
            print(tensor)
        tensor_list.append(tensor_list[-1] + c)
        return tensor_list

    tensor_list = np_tensor_list()
    print("tensor_list:", tensor_list)
    assert len(tensor_list) == 3


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_list_count():
    """
    Feature: Fallback feature
    Description: support attr/method of builtin type.
    Expectation: No exception.
    """
    @ms_function
    def list_count():
        x = list([1, 2, 3])
        res = x.count(1)
        return res
    assert list_count() == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_list_append():
    """
    Feature: Fallback feature
    Description: support attr/method of builtin type.
    Expectation: No exception.
    """
    @ms_function
    def list_append():
        x = list([1, 2, 3])
        x.append(4)
        return Tensor(x)
    assert np.all(list_append().asnumpy() == np.array([1, 2, 3, 4]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_list_insert_1():
    """
    Feature: Fallback feature
    Description: support attr/method of builtin type.
    Expectation: No exception.
    """
    @ms_function
    def list_insert():
        x = list([1, 3, 4])
        x.insert(0, 2)
        return Tensor(x)
    assert np.all(list_insert().asnumpy() == np.array([2, 1, 3, 4]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_list_insert_2():
    """
    Feature: Fallback feature
    Description: support attr/method of builtin type.
    Expectation: No exception.
    """
    @ms_function
    def list_insert():
        x = list([1, 3, 4])
        x.insert(5, 2)
        return Tensor(x)
    assert np.all(list_insert().asnumpy() == np.array([1, 3, 4, 2]))


@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_list_insert_3():
    """
    Feature: Fallback feature
    Description: support attr/method of builtin type.
    Expectation: No exception.
    """
    @ms_function
    def list_insert():
        x = list([1, 3, 4])
        x.insert(-1, 2)
        return Tensor(x)
    assert np.all(list_insert().asnumpy() == np.array([1, 3, 2, 4]))


@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_list_insert_4():
    """
    Feature: Fallback feature
    Description: support attr/method of builtin type.
    Expectation: No exception.
    """
    @ms_function
    def list_insert():
        x = list([1, 3, 4])
        x.insert(-5, 2)
        return Tensor(x)
    assert np.all(list_insert().asnumpy() == np.array([2, 1, 3, 4]))


@ms_function
def np_fallback_func_tensor_index(x):
    array_x = tuple([2, 3, 4, 5])
    np_x = np.array(array_x).astype(np.float32)
    me_x = Tensor(np_x)
    me_x = me_x + me_x
    return me_x[x]


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_fallback_func_tensor_index():
    """
    Feature: Fallback feature: support Tensor index.
    Description: Fallback feature: support Tensor index.
    Expectation: Fallback feature: support Tensor index.
    """
    x = Tensor(1, mstype.int32)
    output = np_fallback_func_tensor_index(x)
    output_expect = Tensor(6, mstype.float32)
    assert output == output_expect


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_calculate():
    """
    Feature: Fallback feature.
    Description: Support numpy calculation.
    Expectation: No exception.
    """
    @ms_function
    def np_calculate():
        x = np.array([3, 1, 2, 4, 5])
        y = x % 2
        z = Tensor(y)
        return z
    assert np.all(np_calculate().asnumpy() == np.array([1, 1, 0, 0, 1]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_tensor_array_astype():
    """
    Feature: JIT Fallback
    Description: Test Tensor(array) with astype() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        me_x = Tensor([1.1, -2.1]).astype("float32")
        return me_x
    print(foo())
