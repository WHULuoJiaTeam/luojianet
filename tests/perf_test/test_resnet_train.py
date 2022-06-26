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

"""Resnet test."""

import numpy as np

import luojianet_ms.context as context
from luojianet_ms import Tensor
from luojianet_ms.common.api import _cell_graph_executor
from .resnet_example import resnet50
from ..train_step_wrap import train_step_with_loss_warp

context.set_context(mode=context.GRAPH_MODE)


def test_train_step():
    net = train_step_with_loss_warp(resnet50())
    net.set_train()
    inp = Tensor(np.ones([1, 3, 224, 224], np.float32))
    label = Tensor(np.zeros([1, 10], np.float32))
    _cell_graph_executor.compile(net, inp, label)
