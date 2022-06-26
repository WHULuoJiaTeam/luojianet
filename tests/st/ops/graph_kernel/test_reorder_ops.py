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
import pytest
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.ops import operations as P


class CastUpNet(nn.Cell):
    def __init__(self):
        super(CastUpNet, self).__init__()
        self.cast = P.Cast()
        self.transpose = P.Transpose()
        self.neg = P.Neg()

    def construct(self, i0):
        res = self.cast(i0, mstype.float32)
        res = self.transpose(res, (1, 0))
        res = self.neg(res)
        return res


def get_castup_output(x0, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = CastUpNet()
    output = net(x0)
    return output


def test_castup():
    x0 = Tensor(np.random.normal(0, 1, (16, 16)).astype(np.float16))
    expect = get_castup_output(x0, False)
    output = get_castup_output(x0, True)
    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()
    assert np.allclose(expect_np, output_np, 1e-4, 1e-4)


class CastDownNet(nn.Cell):
    def __init__(self):
        super(CastDownNet, self).__init__()
        self.cast = P.Cast()
        self.transpose = P.Transpose()
        self.neg = P.Neg()

    def construct(self, i0):
        res = self.transpose(i0, (1, 0))
        res = self.neg(res)
        res = self.cast(res, mstype.float16)
        return res


def get_castdown_output(x0, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = CastDownNet()
    output = net(x0)
    return output


def test_castdown():
    x0 = Tensor(np.random.normal(0, 1, (16, 16)).astype(np.float32))
    expect = get_castdown_output(x0, False)
    output = get_castdown_output(x0, True)
    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()
    assert np.allclose(expect_np, output_np, 1e-3, 1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_castup_gpu():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_castup()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_castup_ascend():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_castup()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_castdown_gpu():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_castdown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_castdown_ascend():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_castdown()
