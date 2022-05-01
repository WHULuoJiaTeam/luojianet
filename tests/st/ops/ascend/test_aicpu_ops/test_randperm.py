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
import luojianet_ms
import luojianet_ms.nn as nn
import luojianet_ms.context as context
from luojianet_ms import Tensor
from luojianet_ms.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Module):
    def __init__(self, max_length, pad, dtype=luojianet_ms.int32):
        super(Net, self).__init__()
        self.randperm = P.Randperm(max_length, pad, dtype)

    def call(self, n):
        return self.randperm(n)


def test_net():
    net = Net(max_length=1, pad=-1)
    output = net(Tensor([1], luojianet_ms.int32))

    print(output)
    print(output.shape)
    print(output.dtype)
    assert output.shape == (1,)
    assert output.dtype == luojianet_ms.int32
    assert output.asnumpy()[0] == 0


def test_net_n20():
    net = Net(max_length=30, pad=-1, dtype=luojianet_ms.int32)
    output = net(Tensor([20], dtype=luojianet_ms.int32))

    print(output)
    assert output.shape == (30,)
    assert output.dtype == luojianet_ms.int32

    sample_set = set()
    for index, i in enumerate(output.asnumpy()):
        if index < 20:
            assert i not in sample_set
            assert 0 <= i < 20
            sample_set.add(i)
        else:
            assert i == -1
