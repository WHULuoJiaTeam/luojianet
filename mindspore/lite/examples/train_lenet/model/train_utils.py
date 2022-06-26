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
"""train_utils."""

import mindspore.nn as nn
from mindspore.common.parameter import ParameterTuple
from mindspore import amp


def train_wrap(net, loss_fn=None, optimizer=None, weights=None, loss_scale_manager=None):
    """
    train_wrap
    """
    if loss_fn is None:
        loss_fn = nn.SoftmaxCrossEntropyWithLogits(reduction='mean', sparse=True)
    loss_net = nn.WithLossCell(net, loss_fn)
    loss_net.set_train()
    if weights is None:
        weights = ParameterTuple(net.trainable_params())
    if optimizer is None:
        optimizer = nn.Adam(weights, learning_rate=0.003, beta1=0.9, beta2=0.999, eps=1e-5, use_locking=False,
                            use_nesterov=False, weight_decay=4e-5, loss_scale=1.0)
    if loss_scale_manager is None:
        train_net = nn.TrainOneStepCell(loss_net, optimizer)
    else:
        train_net = amp.build_train_network(net, optimizer, loss_fn, level="O2", loss_scale_manager=loss_scale_manager)
    return train_net
