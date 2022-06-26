# Copyright 2022 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest

from mindspore import context, Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P

from parallel.utils.utils import compile_net

x_ = Tensor(np.random.normal(size=[32, 8, 8]).astype(np.float32))
input_v_ = Tensor(np.random.normal(size=[16, 8, 8]).astype(np.float32))
indices_ = tuple(range(16))


class InplaceAddNet(Cell):
    def __init__(self, indices, strategy=None):
        super(InplaceAddNet, self).__init__()
        self.inplace_add = P.InplaceAdd(indices).shard(strategy)

    def construct(self, x, input_v):
        return self.inplace_add(x, input_v)


class InplaceSubNet(Cell):
    def __init__(self, indices, strategy=None):
        super(InplaceSubNet, self).__init__()
        self.inplace_sub = P.InplaceSub(indices).shard(strategy)

    def construct(self, x, input_v):
        return self.inplace_sub(x, input_v)


def test_inplace_add_auto_parallel():
    """
    Feature: test InplaceAdd auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = InplaceAddNet(indices_)
    compile_net(net, x_, input_v_)


def test_inplace_add_model_parallel():
    """
    Feature: test InplaceAdd model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 4, 2), (1, 4, 2))
    net = InplaceAddNet(indices_, strategy)
    compile_net(net, x_, input_v_)


def test_inplace_add_model_parallel_with_repeated_cal():
    """
    Feature: test InplaceAdd model parallel with repeated calculation
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 2, 2), (1, 2, 2))
    net = InplaceAddNet(indices_, strategy)
    compile_net(net, x_, input_v_)


def test_inplace_add_strategy_error():
    """
    Feature: test invalid strategy for InplaceAdd
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 4, 2), (1, 2, 4))
    net = InplaceAddNet(indices_, strategy)
    with pytest.raises(RuntimeError):
        compile_net(net, x_, input_v_)


def test_inplace_sub_auto_parallel():
    """
    Feature: test InplaceSub auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = InplaceSubNet(indices_)
    compile_net(net, x_, input_v_)


def test_inplace_sub_model_parallel():
    """
    Feature: test InplaceSub model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 4, 2), (1, 4, 2))
    net = InplaceSubNet(indices_, strategy)
    compile_net(net, x_, input_v_)


def test_inplace_sub_model_parallel_with_repeated_cal():
    """
    Feature: test InplaceSub model parallel with repeated calculation
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 2, 2), (1, 2, 2))
    net = InplaceSubNet(indices_, strategy)
    compile_net(net, x_, input_v_)


def test_inplace_sub_strategy_error():
    """
    Feature: test invalid strategy for InplaceSub
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 4, 2), (1, 2, 4))
    net = InplaceSubNet(indices_, strategy)
    with pytest.raises(RuntimeError):
        compile_net(net, x_, input_v_)
