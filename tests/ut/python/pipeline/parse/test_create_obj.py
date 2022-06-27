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
"""
@File  : test_create_obj.py
@Author:
@Date  : 2019-06-26
@Desc  : test create object instance on parse function, eg: 'forward'
         Support class : nn.Module ops.Primitive
         Support parameter: type is define on function 'ValuePtrToPyData'
                            (int,float,string,bool,tensor)
"""
import logging
import numpy as np
import pytest

import luojianet_ms.nn as nn
from luojianet_ms import context, ops, dtype
from luojianet_ms.common.api import ms_function
from luojianet_ms.common import Tensor, Parameter
from luojianet_ms.ops import operations as P
from ...ut_filter import non_graph_engine

log = logging.getLogger("test")
log.setLevel(level=logging.ERROR)


class Net(nn.Module):
    """ Net definition """

    def __init__(self):
        super(Net, self).__init__()
        self.softmax = nn.Softmax(0)
        self.axis = 0

    def forward(self, x):
        x = nn.Softmax(self.axis)(x)
        return x


# Test: Create Module OR Primitive instance on forward
@non_graph_engine
def test_create_cell_object_on_construct():
    """ test_create_cell_object_on_construct """
    log.debug("begin test_create_object_on_construct")
    context.set_context(mode=context.GRAPH_MODE)
    np1 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(np1)

    net = Net()
    output = net(input_me)
    out_me1 = output.asnumpy()
    print(np1)
    print(out_me1)
    log.debug("finished test_create_object_on_construct")


# Test: Create Module OR Primitive instance on forward
class Net1(nn.Module):
    """ Net1 definition """

    def __init__(self):
        super(Net1, self).__init__()
        self.add = P.Add()

    @ms_function
    def forward(self, x, y):
        add = P.Add()
        result = add(x, y)
        return result


@non_graph_engine
def test_create_primitive_object_on_construct():
    """ test_create_primitive_object_on_construct """
    log.debug("begin test_create_object_on_construct")
    x = Tensor(np.array([[1, 2, 3], [1, 2, 3]], np.float32))
    y = Tensor(np.array([[2, 3, 4], [1, 1, 2]], np.float32))

    net = Net1()
    net.forward(x, y)
    log.debug("finished test_create_object_on_construct")


# Test: Create Module OR Primitive instance on forward use many parameter
class NetM(nn.Module):
    """ NetM definition """

    def __init__(self, name, axis):
        super(NetM, self).__init__()
        # self.relu = nn.ReLU()
        self.name = name
        self.axis = axis
        self.softmax = nn.Softmax(self.axis)

    def forward(self, x):
        x = self.softmax(x)
        return x


class NetC(nn.Module):
    """ NetC definition """

    def __init__(self, tensor):
        super(NetC, self).__init__()
        self.tensor = tensor

    def forward(self, x):
        x = NetM("test", 1)(x)
        return x


# Test: Create Module OR Primitive instance on forward
@non_graph_engine
def test_create_cell_object_on_construct_use_many_parameter():
    """ test_create_cell_object_on_construct_use_many_parameter """
    log.debug("begin test_create_object_on_construct")
    context.set_context(mode=context.GRAPH_MODE)
    np1 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(np1)

    net = NetC(input_me)
    output = net(input_me)
    out_me1 = output.asnumpy()
    print(np1)
    print(out_me1)
    log.debug("finished test_create_object_on_construct")


class NetD(nn.Module):
    """ NetD definition """

    def forward(self, x, y):
        concat = P.Concat(axis=1)
        return concat((x, y))


# Test: Create Module OR Primitive instance on forward
@non_graph_engine
def test_create_primitive_object_on_construct_use_kwargs():
    """ test_create_primitive_object_on_construct_use_kwargs """
    log.debug("begin test_create_primitive_object_on_construct_use_kwargs")
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
    y = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
    net = NetD()
    net(x, y)
    log.debug("finished test_create_primitive_object_on_construct_use_kwargs")


class NetE(nn.Module):
    """ NetE definition """

    def __init__(self):
        super(NetE, self).__init__()
        self.w = Parameter(Tensor(np.ones([16, 16, 3, 3]).astype(np.float32)), name='w')

    def forward(self, x):
        out_channel = 16
        kernel_size = 3
        conv2d = P.Conv2D(out_channel,
                          kernel_size,
                          1,
                          pad_mode='valid',
                          pad=0,
                          stride=1,
                          dilation=1,
                          group=1)
        return conv2d(x, self.w)


# Test: Create Module OR Primitive instance on forward
@non_graph_engine
def test_create_primitive_object_on_construct_use_args_and_kwargs():
    """ test_create_primitive_object_on_construct_use_args_and_kwargs """
    log.debug("begin test_create_primitive_object_on_construct_use_args_and_kwargs")
    context.set_context(mode=context.GRAPH_MODE)
    inputs = Tensor(np.ones([1, 16, 16, 16]).astype(np.float32))
    net = NetE()
    net(inputs)
    log.debug("finished test_create_primitive_object_on_construct_use_args_and_kwargs")


# Test: Create Module instance in forward
class SubCell(nn.Module):
    def __init__(self, t):
        super(SubCell, self).__init__()
        self.t = t

    def forward(self):
        return ops.typeof(self.t)


class WrapCell(nn.Module):
    def forward(self, t):
        type_0 = ops.typeof(t)
        type_1 = SubCell(t)()
        return type_0, type_1


def test_create_cell_with_tensor():
    """
    Feature: Raise exception while create Module(that init use tensor input) in forward.
    Description: None
    Expectation: TypeError.
    """
    t = Tensor(np.zeros((2, 2), np.float), dtype.float32)
    with pytest.raises(TypeError):
        print(WrapCell()(t))
