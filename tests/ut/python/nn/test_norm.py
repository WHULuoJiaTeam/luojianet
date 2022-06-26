# Copyright 2020 Huawei Technologies Co., Ltd
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
""" test norm """
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import _cell_graph_executor
from ..ut_filter import non_graph_engine


class NormNet(nn.Cell):
    def __init__(self):
        super(NormNet, self).__init__()
        self.norm = nn.Norm()

    def construct(self, x):
        return self.norm(x)


@non_graph_engine
def test_compile_norm():
    net = NormNet()
    x = Tensor(np.array([2.0, 1.0]))
    _cell_graph_executor.compile(net, x)
