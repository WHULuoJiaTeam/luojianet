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
import numpy as np

import luojianet_ms.context as context
import luojianet_ms.nn as nn
from luojianet_ms import Tensor
from luojianet_ms.common.api import ms_function
from luojianet_ms.common.initializer import initializer
from luojianet_ms.common.parameter import Parameter
from luojianet_ms.ops import operations as P
from luojianet_ms.ops.composite import GradOperation

context.set_context(device_target="Ascend")


class Grad(nn.Module):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    @ms_function
    def call(self, input_, output_grad):
        return self.grad(self.network)(input_, output_grad)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        out_channel = 512
        kernel_size = 2048
        self.conv = P.Conv2D(out_channel,
                             (kernel_size, kernel_size),
                             mode=1,
                             pad_mode="same",
                             pad=3,
                             stride=2,
                             dilation=1,
                             group=1)
        self.w = Parameter(initializer(
            'normal', [512, 2048, 1, 1]), name='w')

    @ms_function
    def call(self, x):
        return self.conv(x, self.w)


def test_net():
    x = np.ones([32, 2048, 7, 7]).astype(np.float32)
    sens = np.ones([32, 512, 7, 7]).astype(np.float32)
    net = Grad(Net())
    output = net(Tensor(x), Tensor(sens))
    print(output.asnumpy())
