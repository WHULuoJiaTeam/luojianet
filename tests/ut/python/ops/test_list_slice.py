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
""" test_list_slice """
import numpy as np

import luojianet_ms.ops.operations as P
from luojianet_ms import Tensor
from luojianet_ms.nn import Module
from ....luojianet_ms_test_framework.luojianet_ms_test import luojianet_ms_test
from ....luojianet_ms_test_framework.pipeline.forward.compile_forward \
    import pipeline_for_compile_forward_ge_graph_for_case_by_case_config_input_list
from ....luojianet_ms_test_framework.pipeline.forward.verify_exception \
    import pipeline_for_verify_exception_for_case_by_case_config


class NetWork_1(Module):
    """ NetWork_1 definition """

    def __init__(self):
        super(NetWork_1, self).__init__()
        self.addN = P.AddN()
        self.index_0 = Tensor(3)
        self.index_1 = Tensor([5])
        self.index_3 = Tensor([True])

    def forward(self, tensor_list):
        tensor_list_slice0 = tensor_list[:]
        tensor_list_slice1 = tensor_list[:self.index_0]
        tensor_list_slice2 = tensor_list[self.index_3:]
        tensor_list_slice3 = tensor_list[2:self.index_1:True]
        sum0 = self.addN(tensor_list_slice0)
        sum1 = self.addN(tensor_list_slice1)
        sum2 = self.addN(tensor_list_slice2)
        sum3 = self.addN(tensor_list_slice3)
        ret = sum0 + sum1 + sum2 + sum3
        return ret


class NetWork_2(Module):
    """ NetWork_2 definition """

    def __init__(self):
        super(NetWork_2, self).__init__()
        self.addN = P.AddN()
        self.step = Tensor([-1])
        self.index_0 = Tensor(-6)

    def forward(self, tensor_list):
        tensor_list_slice0 = tensor_list[::self.step]
        tensor_list_slice1 = tensor_list[-1::-1]
        tensor_list_slice2 = tensor_list[:-4:-1]
        tensor_list_slice3 = tensor_list[self.index_0:3]
        tensor_list_slice4 = tensor_list[-1:-6:-2]
        sum0 = self.addN(tensor_list_slice0)
        sum1 = self.addN(tensor_list_slice1)
        sum2 = self.addN(tensor_list_slice2)
        sum3 = self.addN(tensor_list_slice3)
        sum4 = self.addN(tensor_list_slice4)
        ret = sum0 + sum1 + sum2 + sum3 + sum4
        return ret


class NetWorkSliceStepZero(Module):
    """ NetWorkSliceStepZero definition """

    def __init__(self):
        super(NetWorkSliceStepZero, self).__init__()
        self.start = 0
        self.stop = 3
        self.step = 0

    def forward(self, tensor_list):
        tensor_list_slice = tensor_list[self.start:self.stop:self.step]
        return tensor_list_slice


class NetWorkOutOfBounds(Module):
    """ NetWork_3 definition """

    def __init__(self):
        super(NetWorkOutOfBounds, self).__init__()
        self.index = 100

    def forward(self, tensor_list):
        return tensor_list[self.index]


class NetWorkTensorSizeGreaterThanTwo(Module):
    """ NetWork_3 definition """

    def __init__(self):
        super(NetWorkTensorSizeGreaterThanTwo, self).__init__()
        self.index_0 = Tensor([2, 3])

    def forward(self, tensor_list):
        return tensor_list[1:self.index_0]


class NetWorkTensorDtypeFloat(Module):
    """ NetWork_3 definition """

    def __init__(self):
        super(NetWorkTensorDtypeFloat, self).__init__()
        self.index_0 = Tensor([2.1])

    def forward(self, tensor_list):
        return tensor_list[1:self.index_0]

class NewWorkSliceVarTensorError(Module):
    """ error Network definition """

    def __init__(self):
        super(NewWorkSliceVarTensorError, self).__init__()
        self.index_0 = Tensor(2)

    def forward(self, tensor_list, y):
        x = y + self.index_0
        return tensor_list[1:x]

test_cases = [
    ('SlicePositive', {
        'block': NetWork_1(),
        'desc_inputs': [[Tensor(np.ones([2, 3, 4], np.int32)),
                         Tensor(np.zeros([2, 3, 4], np.int32)),
                         Tensor(np.ones([2, 3, 4], np.int32)),
                         Tensor(np.ones([2, 3, 4], np.int32)),
                         Tensor(np.zeros([2, 3, 4], np.int32)),
                         Tensor(np.ones([2, 3, 4], np.int32))]],
    }),
    ('SliceNegative', {
        'block': NetWork_2(),
        'desc_inputs': [[Tensor(np.ones([2, 3, 4], np.int32)),
                         Tensor(np.zeros([2, 3, 4], np.int32)),
                         Tensor(np.ones([2, 3, 4], np.int32)),
                         Tensor(np.ones([2, 3, 4], np.int32)),
                         Tensor(np.zeros([2, 3, 4], np.int32)),
                         Tensor(np.ones([2, 3, 4], np.int32))]],
    }),
]

test_cases_for_verify_exception = [
    ('SliceStepZero', {
        'block': (NetWorkSliceStepZero(), {'exception': ValueError}),
        'desc_inputs': [[Tensor(np.ones([2, 3, 4], np.int32)),
                         Tensor(np.zeros([2, 3, 4], np.int32)),
                         Tensor(np.ones([2, 3, 4], np.int32))]],
    }),
    ('SliceOutOfBounds', {
        'block': (NetWorkOutOfBounds(), {'exception': IndexError}),
        'desc_inputs': [[Tensor(np.ones([2, 3, 4], np.int32)),
                         Tensor(np.zeros([2, 3, 4], np.int32)),
                         Tensor(np.ones([2, 3, 4], np.int32))]],
    }),
    ('SliceTensorSizeGreaterThanTwo', {
        'block': (NetWorkTensorSizeGreaterThanTwo(), {'exception': TypeError}),
        'desc_inputs': [[Tensor(np.ones([2, 3, 4], np.int32)),
                         Tensor(np.zeros([2, 3, 4], np.int32)),
                         Tensor(np.ones([2, 3, 4], np.int32))]],
    }),
    ('SliceTensorDtypeFloat', {
        'block': (NetWorkTensorDtypeFloat(), {'exception': TypeError}),
        'desc_inputs': [[Tensor(np.ones([2, 3, 4], np.int32)),
                         Tensor(np.zeros([2, 3, 4], np.int32)),
                         Tensor(np.ones([2, 3, 4], np.int32))]],
    }),
    ('NewWorkSliceVarTensorError', {
        'block': (NewWorkSliceVarTensorError(), {'exception': TypeError}),
        'desc_inputs': [[Tensor(np.ones([2, 3, 4], np.int32)),
                         Tensor(np.zeros([2, 3, 4], np.int32)),
                         Tensor(np.ones([2, 3, 4], np.int32))], Tensor(1)],
    }),
]


@luojianet_ms_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config_input_list)
def test_compile():
    """
    Feature: test list slice
    Description: test list slice positive and negative
    Expectation: success
    """
    return test_cases


@luojianet_ms_test(pipeline_for_verify_exception_for_case_by_case_config)
def test_check_exception():
    """
    Feature: test list getitem exception
    Description: test list getitem exception
    Expectation: throw errors
    """
    return test_cases_for_verify_exception
