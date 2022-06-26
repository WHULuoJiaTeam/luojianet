# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
from mindspore.common.api import ms_function
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner

x0 = np.random.rand(2, 3, 4, 4).astype(np.float32)
axis0 = 3
keep_dims0 = True

x1 = np.random.rand(2, 3, 4, 4).astype(np.float32)
axis1 = 3
keep_dims1 = False

x2 = np.random.rand(2, 3, 1, 4).astype(np.float32)
axis2 = 2
keep_dims2 = True

x3 = np.random.rand(2, 3, 1, 4).astype(np.float32)
axis3 = 2
keep_dims3 = False

x4 = np.random.rand(2, 3, 4, 4).astype(np.float32)
axis4 = ()
np_axis4 = None
keep_dims4 = True

x5 = np.random.rand(2, 3, 4, 4).astype(np.float32)
axis5 = ()
np_axis5 = None
keep_dims5 = False

x6 = np.random.rand(2, 3, 4, 4).astype(np.float32)
axis6 = (1, 2)
keep_dims6 = False

x7 = np.random.rand(2, 3, 4, 4).astype(np.float32)
axis7 = (1, 2)
keep_dims7 = True

x8 = np.random.rand(2, 1, 1, 4).astype(np.float32)
axis8 = (1, 2, 2, 1, 1, 2)
keep_dims8 = True

x9 = np.random.rand(2, 1, 1, 4).astype(np.float32)
axis9 = (1, 2)
keep_dims9 = False

x10 = np.random.rand(2, 3, 4, 4).astype(np.float32)
axis10 = (0, 1, 2, 3)
keep_dims10 = False

x11 = np.random.rand(1, 1, 1, 1).astype(np.float32)
axis11 = (0, 1, 1, 2, 3, 2, 2, 0, 3)
keep_dims11 = False

x12 = np.random.rand(2, 3, 4, 4).astype(np.float32)
axis12 = -2
keep_dims12 = False

x13 = np.random.rand(2, 3, 4, 4).astype(np.float32)
axis13 = (-2, -1)
keep_dims13 = True

x14 = np.random.rand(1, 1, 1, 1).astype(np.float32)
axis14 = ()
np_axis14 = None
keep_dims14 = True


class ReduceSum(nn.Cell):
    def __init__(self):
        super(ReduceSum, self).__init__()

        self.x0 = Tensor(x0)
        self.axis0 = axis0
        self.keep_dims0 = keep_dims0

        self.x1 = Tensor(x1)
        self.axis1 = axis1
        self.keep_dims1 = keep_dims1

        self.x2 = Tensor(x2)
        self.axis2 = axis2
        self.keep_dims2 = keep_dims2

        self.x3 = Tensor(x3)
        self.axis3 = axis3
        self.keep_dims3 = keep_dims3

        self.x4 = Tensor(x4)
        self.axis4 = axis4
        self.keep_dims4 = keep_dims4

        self.x5 = Tensor(x5)
        self.axis5 = axis5
        self.keep_dims5 = keep_dims5

        self.x6 = Tensor(x6)
        self.axis6 = axis6
        self.keep_dims6 = keep_dims6

        self.x7 = Tensor(x7)
        self.axis7 = axis7
        self.keep_dims7 = keep_dims7

        self.x8 = Tensor(x8)
        self.axis8 = axis8
        self.keep_dims8 = keep_dims8

        self.x9 = Tensor(x9)
        self.axis9 = axis9
        self.keep_dims9 = keep_dims9

        self.x10 = Tensor(x10)
        self.axis10 = axis10
        self.keep_dims10 = keep_dims10

        self.x11 = Tensor(x11)
        self.axis11 = axis11
        self.keep_dims11 = keep_dims11

        self.x12 = Tensor(x12)
        self.axis12 = axis12
        self.keep_dims12 = keep_dims12

        self.x13 = Tensor(x13)
        self.axis13 = axis13
        self.keep_dims13 = keep_dims13

        self.x14 = Tensor(x14)
        self.axis14 = axis14
        self.keep_dims14 = keep_dims14

    @ms_function
    def construct(self):
        return (P.ReduceSum(self.keep_dims0)(self.x0, self.axis0),
                P.ReduceSum(self.keep_dims1)(self.x1, self.axis1),
                P.ReduceSum(self.keep_dims2)(self.x2, self.axis2),
                P.ReduceSum(self.keep_dims3)(self.x3, self.axis3),
                P.ReduceSum(self.keep_dims4)(self.x4, self.axis4),
                P.ReduceSum(self.keep_dims5)(self.x5, self.axis5),
                P.ReduceSum(self.keep_dims6)(self.x6, self.axis6),
                P.ReduceSum(self.keep_dims7)(self.x7, self.axis7),
                P.ReduceSum(self.keep_dims8)(self.x8, self.axis8),
                P.ReduceSum(self.keep_dims9)(self.x9, self.axis9),
                P.ReduceSum(self.keep_dims10)(self.x10, self.axis10),
                P.ReduceSum(self.keep_dims11)(self.x11, self.axis11),
                P.ReduceSum(self.keep_dims12)(self.x12, self.axis12),
                P.ReduceSum(self.keep_dims13)(self.x13, self.axis13),
                P.ReduceSum(self.keep_dims14)(self.x14, self.axis14))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ReduceSum():
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    reduce_sum = ReduceSum()
    output = reduce_sum()

    expect0 = np.sum(x0, axis=axis0, keepdims=keep_dims0)
    diff0 = abs(output[0].asnumpy() - expect0)
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    assert np.all(diff0 < error0)
    assert output[0].shape == expect0.shape

    expect1 = np.sum(x1, axis=axis1, keepdims=keep_dims1)
    diff1 = abs(output[1].asnumpy() - expect1)
    error1 = np.ones(shape=expect1.shape) * 1.0e-5
    assert np.all(diff1 < error1)
    assert output[1].shape == expect1.shape

    expect2 = np.sum(x2, axis=axis2, keepdims=keep_dims2)
    diff2 = abs(output[2].asnumpy() - expect2)
    error2 = np.ones(shape=expect2.shape) * 1.0e-5
    assert np.all(diff2 < error2)
    assert output[2].shape == expect2.shape

    expect3 = np.sum(x3, axis=axis3, keepdims=keep_dims3)
    diff3 = abs(output[3].asnumpy() - expect3)
    error3 = np.ones(shape=expect3.shape) * 1.0e-5
    assert np.all(diff3 < error3)
    assert output[3].shape == expect3.shape

    expect4 = np.sum(x4, axis=np_axis4, keepdims=keep_dims4)
    diff4 = abs(output[4].asnumpy() - expect4)
    error4 = np.ones(shape=expect4.shape) * 1.0e-5
    assert np.all(diff4 < error4)
    assert output[4].shape == expect4.shape

    expect5 = np.sum(x5, axis=np_axis5, keepdims=keep_dims5)
    diff5 = abs(output[5].asnumpy() - expect5)
    error5 = np.ones(shape=expect5.shape) * 1.0e-5
    assert np.all(diff5 < error5)
    assert output[5].shape == expect5.shape

    expect6 = np.sum(x6, axis=axis6, keepdims=keep_dims6)
    diff6 = abs(output[6].asnumpy() - expect6)
    error6 = np.ones(shape=expect6.shape) * 1.0e-5
    assert np.all(diff6 < error6)
    assert output[6].shape == expect6.shape

    expect7 = np.sum(x7, axis=axis7, keepdims=keep_dims7)
    diff7 = abs(output[7].asnumpy() - expect7)
    error7 = np.ones(shape=expect7.shape) * 1.0e-5
    assert np.all(diff7 < error7)
    assert output[7].shape == expect7.shape

    expect8 = np.sum(x8, axis=tuple(set(axis8)), keepdims=keep_dims8)
    diff8 = abs(output[8].asnumpy() - expect8)
    error8 = np.ones(shape=expect8.shape) * 1.0e-5
    assert np.all(diff8 < error8)
    assert output[8].shape == expect8.shape

    expect9 = np.sum(x9, axis=axis9, keepdims=keep_dims9)
    diff9 = abs(output[9].asnumpy() - expect9)
    error9 = np.ones(shape=expect9.shape) * 1.0e-5
    assert np.all(diff9 < error9)
    assert output[9].shape == expect9.shape

    expect10 = np.sum(x10, axis=axis10, keepdims=keep_dims10)
    diff10 = abs(output[10].asnumpy() - expect10)
    error10 = np.ones(shape=expect10.shape) * 1.0e-5
    assert np.all(diff10 < error10)
    assert output[10].shape == expect10.shape

    expect11 = np.sum(x11, axis=tuple(set(axis11)), keepdims=keep_dims11)
    diff11 = abs(output[11].asnumpy() - expect11)
    error11 = np.ones(shape=expect11.shape) * 1.0e-5
    assert np.all(diff11 < error11)
    assert output[11].shape == expect11.shape

    expect12 = np.sum(x12, axis=axis12, keepdims=keep_dims12)
    diff12 = abs(output[12].asnumpy() - expect12)
    error12 = np.ones(shape=expect12.shape) * 1.0e-5
    assert np.all(diff12 < error12)
    assert output[12].shape == expect12.shape

    expect13 = np.sum(x13, axis=axis13, keepdims=keep_dims13)
    diff13 = abs(output[13].asnumpy() - expect13)
    error13 = np.ones(shape=expect13.shape) * 1.0e-5
    assert np.all(diff13 < error13)
    assert output[13].shape == expect13.shape

    expect14 = np.sum(x14, axis=np_axis14, keepdims=keep_dims14)
    diff14 = abs(output[14].asnumpy() - expect14)
    error14 = np.ones(shape=expect14.shape) * 1.0e-5
    assert np.all(diff14 < error14)
    assert output[14].shape == expect14.shape


x_1 = x8
axis_1 = 0
x_2 = x1
axis_2 = 0


class ReduceSumDynamic(nn.Cell):
    def __init__(self, x, axis):
        super(ReduceSumDynamic, self).__init__()
        self.reducesum = P.ReduceSum(True)
        self.test_dynamic = inner.GpuConvertToDynamicShape()
        self.x = x
        self.axis = axis

    def construct(self):
        dynamic_x = self.test_dynamic(self.x)
        return self.reducesum(dynamic_x, self.axis)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reduce_sum_dynamic():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net1 = ReduceSumDynamic(Tensor(x_1), axis_1)
    net2 = ReduceSumDynamic(Tensor(x_2), axis_2)

    expect_1 = np.sum(x_1, axis=axis_1, keepdims=True)
    expect_2 = np.sum(x_2, axis=axis_2, keepdims=True)

    output1 = net1()
    output2 = net2()

    np.testing.assert_almost_equal(output1.asnumpy(), expect_1)
    np.testing.assert_almost_equal(output2.asnumpy(), expect_2)


class ReduceSumTypeNet(nn.Cell):
    def __init__(self, nptype):
        super(ReduceSumTypeNet, self).__init__()
        self.x0 = Tensor(x0.astype(nptype))
        self.axis0 = axis0
        self.keep_dims0 = keep_dims0

    def construct(self):
        return P.ReduceSum(self.keep_dims0)(self.x0, self.axis0)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reduce_sum_float64():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = ReduceSumTypeNet(np.float64)
    output = net()
    expect = np.sum(x0, axis=axis0, keepdims=keep_dims0).astype(np.float64)
    diff = abs(output.asnumpy() - expect)
    error = np.ones(shape=expect.shape) * 1.0e-5
    assert np.all(diff < error)
    assert output.shape == expect.shape

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = ReduceSumTypeNet(np.float64)
    output = net()
    expect = np.sum(x0, axis=axis0, keepdims=keep_dims0).astype(np.float64)
    diff = abs(output.asnumpy() - expect)
    error = np.ones(shape=expect.shape) * 1.0e-5
    assert np.all(diff < error)
    assert output.shape == expect.shape
