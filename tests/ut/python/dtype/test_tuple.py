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
import functools
import numpy as np

import luojianet_ms.context as context
import luojianet_ms.nn as nn
from luojianet_ms import Tensor
from luojianet_ms import dtype as mstype
from tests.ut.python.ut_filter import non_graph_engine
from tests.luojianet_ms_test_framework.luojianet_ms_test import luojianet_ms_test
from tests.luojianet_ms_test_framework.pipeline.forward.compile_forward \
    import pipeline_for_compile_forward_ge_graph_for_case_by_case_config

context.set_context(mode=context.GRAPH_MODE)


class TupleGraphNet(nn.Module):
    def __init__(self):
        super(TupleGraphNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, 3, pad_mode='same')
        self.conv2 = nn.Conv2d(3, 1, 7, pad_mode='same')
        self.conv3 = nn.Conv2d(3, 3, 3, pad_mode='same')
        self.layers = (self.conv1, self.conv2, self.conv3)

    def forward(self, x):
        return self.layers[0](x)


class NestTupleGraphNet(nn.Module):
    def __init__(self):
        super(NestTupleGraphNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, 3, pad_mode='same')
        self.conv2 = nn.Conv2d(3, 1, 7, pad_mode='same')
        self.conv3 = nn.Conv2d(3, 3, 3, pad_mode='same')
        self.layers = ((self.conv1, self.conv2),
                       (self.conv2, self.conv1, self.conv3))

    def forward(self, x):
        return self.layers[0][1](x)


class InTupleNet(nn.Module):
    def __init__(self):
        super(InTupleNet, self).__init__()
        self.tuple_ = (1, 2, 3, 4, 5, "ok")

    def forward(self, x):
        ret = x
        if 2 in self.tuple_:
            ret = x + x
            if "ok" in self.tuple_:
                ret = x - x
        return ret


class TensorInTuple(nn.Module):
    def __init__(self):
        super(TensorInTuple, self).__init__()
        self.t1 = Tensor(1, mstype.float32)
        self.t2 = Tensor(2, mstype.float32)
        self.tuple_ = (self.t1, self.t2)

    def forward(self, x):
        ret = x
        if self.t1 in self.tuple_:
            ret = x + x
        return ret


class TensorNotInTuple(nn.Module):
    def __init__(self):
        super(TensorNotInTuple, self).__init__()
        self.t1 = Tensor(1, mstype.float32)
        self.t2 = Tensor(2, mstype.float32)
        self.tuple_ = (self.t1, self.t2)

    def forward(self, x):
        ret = x
        if self.t1 not in self.tuple_:
            ret = x + x
        return ret


test_case_ops = [
    ('TupleGraph', {
        'block': TupleGraphNet(),
        'desc_inputs': [Tensor(np.ones((3, 3, 24, 24)), mstype.float32)]}),
    ('NestTupleGraph', {
        'block': NestTupleGraphNet(),
        'desc_inputs': [Tensor(np.ones((3, 3, 24, 24)), mstype.float32)]}),
    ('InTuple', {
        'block': InTupleNet(),
        'desc_inputs': [Tensor(np.ones((3, 3, 24, 24)), mstype.float32)]}),
    ('TensorInTuple', {
        'block': TensorInTuple(),
        'desc_inputs': [Tensor(np.ones((3, 3, 24, 24)), mstype.float32)]}),
    ('TensorNotInTuple', {
        'block': TensorNotInTuple(),
        'desc_inputs': [Tensor(np.ones((3, 3, 24, 24)), mstype.float32)]}),
]

test_case_lists = [test_case_ops]
test_exec_case = functools.reduce(lambda x, y: x + y, test_case_lists)


# use -k to select certain testcast
# pytest tests/python/ops/test_ops.py::test_backward -k LayerNorm


@non_graph_engine
@luojianet_ms_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config)
def test_exec():
    context.set_context(mode=context.GRAPH_MODE)
    return test_exec_case
