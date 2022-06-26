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
"""nin_train_export."""

import sys
import numpy as np
from train_utils import save_inout, train_wrap
from NetworkInNetwork import NiN
import mindspore.common.dtype as mstype
from mindspore import context, Tensor, nn
from mindspore.train.serialization import export

context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)

n = NiN(num_classes=10)
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
optimizer = nn.SGD(n.trainable_params(), learning_rate=0.01, momentum=0.9, dampening=0.0, weight_decay=5e-4,
                   nesterov=True, loss_scale=0.9)
net = train_wrap(n, loss_fn, optimizer)

batch = 2
x = Tensor(np.random.randn(batch, 3, 32, 32), mstype.float32)
label = Tensor(np.zeros([batch]).astype(np.int32))
export(net, x, label, file_name="mindir/nin_train", file_format='MINDIR')

if len(sys.argv) > 1:
    save_inout(sys.argv[1] + "nin", x, label, n, net)
