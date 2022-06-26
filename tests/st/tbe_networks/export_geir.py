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
import numpy as np
from resnet_torch import resnet50

from mindspore import Tensor
from mindspore.train.serialization import context, export

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


def test_resnet50_export(batch_size=1, num_classes=5):
    input_np = np.random.uniform(0.0, 1.0, size=[batch_size, 3, 224, 224]).astype(np.float32)
    net = resnet50(batch_size, num_classes)
    export(net, Tensor(input_np), file_name="./me_resnet50.pb", file_format="AIR")
