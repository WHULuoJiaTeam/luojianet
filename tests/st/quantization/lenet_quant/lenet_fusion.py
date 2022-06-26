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
"""LeNet."""
import mindspore.nn as nn


class LeNet5(nn.Cell):
    """
    Lenet network

    Args:
        num_class (int): Num classes. Default: 10.

    Returns:
        Tensor, output tensor
    Examples:
        >>> LeNet(num_class=10)

    """

    def __init__(self, num_class=10, channel=1):
        super(LeNet5, self).__init__()
        self.type = "fusion"
        self.num_class = num_class

        # change `nn.Conv2d` to `nn.Conv2dBnAct`
        self.conv1 = nn.Conv2dBnAct(channel, 6, 5, pad_mode='valid', activation='relu')
        self.conv2 = nn.Conv2dBnAct(6, 16, 5, pad_mode='valid', activation='relu')
        # change `nn.Dense` to `nn.DenseBnAct`
        self.fc1 = nn.DenseBnAct(16 * 5 * 5, 120, activation='relu')
        self.fc2 = nn.DenseBnAct(120, 84, activation='relu')
        self.fc3 = nn.DenseBnAct(84, self.num_class)

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
