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

import os
import random
import numpy as np
from resnet import resnet50

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as vision
import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore import Tensor
from mindspore import context
from mindspore.nn.optim.momentum import Momentum
from mindspore.ops import operations as P
from mindspore.train.callback import Callback
from mindspore.train.model import Model

random.seed(1)
np.random.seed(1)
ds.config.set_seed(1)

data_home = "/home/workspace/mindspore_dataset"


def create_dataset(repeat_num=1, training=True, batch_size=32):
    data_dir = data_home + "/cifar-10-batches-bin"
    if not training:
        data_dir = data_home + "/cifar-10-verify-bin"
    data_set = ds.Cifar10Dataset(data_dir)

    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    random_crop_op = vision.RandomCrop(
        (32, 32), (4, 4, 4, 4))  # padding_mode default CONSTANT
    random_horizontal_op = vision.RandomHorizontalFlip()
    # interpolation default BILINEAR
    resize_op = vision.Resize((resize_height, resize_width))
    rescale_op = vision.Rescale(rescale, shift)
    normalize_op = vision.Normalize(
        (0.4465, 0.4822, 0.4914), (0.2010, 0.1994, 0.2023))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op,
                changeswap_op]

    # apply map operations on images
    data_set = data_set.map(operations=type_cast_op, input_columns="label")
    data_set = data_set.map(operations=c_trans, input_columns="image")

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=1000)

    # apply batch operations
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    # apply repeat operations
    data_set = data_set.repeat(repeat_num)

    return data_set


class CrossEntropyLoss(nn.Cell):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy = P.SoftmaxCrossEntropyWithLogits()
        self.mean = P.ReduceMean()
        self.one_hot = P.OneHot()
        self.one = Tensor(1.0, mstype.float32)
        self.zero = Tensor(0.0, mstype.float32)

    def construct(self, logits, label):
        label = self.one_hot(label, F.shape(logits)[1], self.one, self.zero)
        loss = self.cross_entropy(logits, label)[0]
        loss = self.mean(loss, (-1,))
        return loss


class LossGet(Callback):
    def __init__(self, per_print_times=1):
        super(LossGet, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self._loss = 0.0

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training."
                             .format(cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            self._loss = loss
            print("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num, cur_step_in_epoch, loss))

    def get_loss(self):
        return self._loss


def train_process(epoch_size, num_classes, batch_size):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    net = resnet50(batch_size, num_classes)
    loss = CrossEntropyLoss()
    opt = Momentum(filter(lambda x: x.requires_grad,
                          net.get_parameters()), 0.01, 0.9)

    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

    dataset = create_dataset(1, training=True, batch_size=batch_size)
    loss_cb = LossGet()
    model.train(epoch_size, dataset, callbacks=[loss_cb])

    net.set_train(False)
    eval_dataset = create_dataset(1, training=False)
    res = model.eval(eval_dataset)
    print("result: ", res)
    return res


def test_resnet_cifar_1p():
    epoch_size = 1
    num_classes = 10
    batch_size = 32
    acc = train_process(epoch_size, num_classes, batch_size)
    os.system("rm -rf kernel_meta")
    print("End training...")
    assert acc['acc'] > 0.20
