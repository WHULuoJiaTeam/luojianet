# Copyright 2021 Huawei Technologies Co., Ltd
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
"""shufflenetv2_train_export."""

import sys
import numpy as np
from train_utils import save_inout, train_wrap
from official.cv.shufflenetv2.src.shufflenetv2 import ShuffleNetV2
import mindspore.common.dtype as mstype
from mindspore import context, Tensor, nn
from mindspore.train.serialization import export

context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)

n = ShuffleNetV2(n_class=10)
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
optimizer = nn.Momentum(n.trainable_params(), 0.01, 0.9, use_nesterov=False)

net = train_wrap(n, loss_fn, optimizer)

batch = 2
x = Tensor(np.random.randn(batch, 3, 224, 224), mstype.float32)
label = Tensor(np.zeros([batch, 10]).astype(np.float32))
export(net, x, label, file_name="mindir/shufflenetv2_train", file_format='MINDIR')

if len(sys.argv) > 1:
    save_inout(sys.argv[1] + "shufflenetv2", x, label, n, net)
