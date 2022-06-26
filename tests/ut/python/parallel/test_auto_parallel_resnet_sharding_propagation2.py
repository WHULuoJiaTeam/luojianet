# Copyright 2019 Huawei Technologies Co., Ltd
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

import re
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.common.initializer import TruncatedNormal
from mindspore.communication.management import init
from mindspore.nn.loss.loss import _Loss
from mindspore.nn.optim.momentum import Momentum
from mindspore.ops import operations as P
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._utils import _reset_op_id as resset_op_id
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.communication._comm_helper import GlobalComm

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
context.set_context.__wrapped__(device_id=0)
GlobalComm.CHECK_ENVS = False
init()
GlobalComm.CHECK_ENVS = True


def weight_variable():
    return TruncatedNormal(0.02)


def _conv3x3(in_channels, out_channels, stride=1, padding=0, pad_mode='same'):
    """Get a conv2d layer with 3x3 kernel size."""
    init_value = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=stride, padding=padding, pad_mode=pad_mode, weight_init=init_value)


def _conv1x1(in_channels, out_channels, stride=1, padding=0, pad_mode='same'):
    """Get a conv2d layer with 1x1 kernel size."""
    init_value = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=1, stride=stride, padding=padding, pad_mode=pad_mode, weight_init=init_value)


def _conv7x7(in_channels, out_channels, stride=1, padding=0, pad_mode='same'):
    """Get a conv2d layer with 7x7 kernel size."""
    init_value = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=7, stride=stride, padding=padding, pad_mode=pad_mode, weight_init=init_value)


def _fused_bn(channels, momentum=0.9):
    """Get a fused batchnorm"""
    return nn.BatchNorm2d(channels, momentum=momentum)


class ResidualBlock(nn.Cell):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 momentum=0.9):
        super(ResidualBlock, self).__init__()

        out_chls = out_channels // self.expansion
        self.conv1 = _conv1x1(in_channels, out_chls, stride=1)
        self.bn1 = _fused_bn(out_chls, momentum=momentum)

        self.conv2 = _conv3x3(out_chls, out_chls, stride=stride)
        self.bn2 = _fused_bn(out_chls, momentum=momentum)

        self.conv3 = _conv1x1(out_chls, out_channels, stride=1)
        self.bn3 = _fused_bn(out_channels, momentum=momentum)

        self.relu = P.ReLU()
        self.downsample = (in_channels != out_channels)
        self.stride = stride
        if self.downsample:
            self.conv_down_sample = _conv1x1(in_channels, out_channels,
                                             stride=stride)
            self.bn_down_sample = _fused_bn(out_channels, momentum=momentum)
        elif self.stride != 1:
            self.maxpool_down = nn.MaxPool2d(kernel_size=1, stride=2, pad_mode='same')

        self.add = P.Add()

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            identity = self.conv_down_sample(identity)
            identity = self.bn_down_sample(identity)
        elif self.stride != 1:
            identity = self.maxpool_down(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides=None,
                 num_classes=100,
                 matmul_stra=None,
                 squeeze_stra=None):
        super(ResNet, self).__init__()
        if strides is None:
            strides = [1, 2, 2, 2]
        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of "
                             "layer_num, inchannel, outchannel list must be 4!")

        self.conv1 = _conv7x7(3, 64, stride=2)
        self.bn1 = _fused_bn(64)
        self.relu = P.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0])
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1])
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2])
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3])

        self.mean = P.ReduceMean(keep_dims=True)
        self.end_point = nn.Dense(2048, num_classes, has_bias=True,
                                  weight_init=weight_variable(),
                                  bias_init=weight_variable()).add_flags_recursive(fp16=True)
        self.end_point.matmul.shard(matmul_stra)
        self.squeeze = P.Squeeze().shard(squeeze_stra)
        self.cast = P.Cast()

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        layers = []

        resblk = block(in_channel, out_channel, stride=1)
        layers.append(resblk)

        for _ in range(1, layer_num - 1):
            resblk = block(out_channel, out_channel, stride=1)
            layers.append(resblk)

        resblk = block(out_channel, out_channel, stride=stride)
        layers.append(resblk)

        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        out = self.mean(c5, (2, 3))
        out = self.squeeze(out)
        out = self.end_point(out)

        return out


def resnet50(class_num=10, matmul_stra=None, squeeze_stra=None):
    return ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [2, 2, 2, 1],
                  class_num,
                  matmul_stra=matmul_stra,
                  squeeze_stra=squeeze_stra)


class SoftmaxCrossEntropyExpand(_Loss):
    def __init__(self, sparse=False):
        super(SoftmaxCrossEntropyExpand, self).__init__()
        self.exp = P.Exp()
        self.sum = P.ReduceSum(keep_dims=True)
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.div = P.Div()
        self.log = P.Log()
        self.sum_cross_entropy = P.ReduceSum(keep_dims=False)
        self.mul = P.Mul()
        self.mul2 = P.Mul()
        self.cast = P.Cast()
        self.mean = P.ReduceMean(keep_dims=False).add_prim_attr("cross_batch", True)
        self.sparse = sparse
        self.max = P.ReduceMax(keep_dims=True).shard(((8, 1),))
        self.sub = P.Sub().shard(((8, 1), (1, 1)))
        self.cast1 = P.Cast()

    def construct(self, logit, label):
        logit = self.cast1(logit, mstype.float32)
        logit_max = self.max(logit)
        exp = self.exp(self.sub(logit, logit_max))
        exp_sum = self.sum(exp, -1)
        softmax_result = self.div(exp, exp_sum)
        if self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)

        softmax_result_log = self.log(softmax_result)
        loss = self.sum_cross_entropy((self.mul(softmax_result_log, label)), -1)
        loss = self.mul2(F.scalar_to_array(-1.0), loss)
        loss = self.mean(loss, -1)

        return loss


class DatasetLenet():
    def __init__(self, predict, label, length=3):
        self.predict = predict
        self.label = label
        self.index = 0
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.predict, self.label

    def reset(self):
        self.index = 0

    def get_dataset_size(self):
        return 32

    def get_repeat_count(self):
        return 1

    def create_tuple_iterator(self, num_epochs=-1, do_copy=True):
        return self


def test_train_32k_8p(batch_size=32, num_classes=32768):
    dev_num = 8
    context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, device_num=dev_num,
                                      search_mode="sharding_propagation")
    set_algo_parameters(elementwise_op_strategy_follow=True)
    resset_op_id()
    np.random.seed(6)
    input_np = np.ones([batch_size, 3, 224, 224]).astype(np.float32)
    label_np = np.zeros([batch_size]).astype(np.int32)
    for i in range(0, batch_size):
        label_np[i] = i % num_classes
    dataset = DatasetLenet(Tensor(input_np), Tensor(label_np), 1)
    matmul_stra = ((dev_num, 1), (1, 1))
    net = resnet50(num_classes, matmul_stra)
    loss = SoftmaxCrossEntropyExpand(sparse=True)
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    model = Model(net, loss_fn=loss, optimizer=opt)
    model.train(5, dataset, dataset_sink_mode=False)
    strategies = _cell_graph_executor._get_shard_strategy(model._train_network)
    for (k, v) in strategies.items():
        if re.search('Conv2D-op', k) is not None:
            assert v[0][0] == dev_num
        elif re.search('MatMul-op', k) is not None:
            assert v == [[dev_num, 1], [1, 1]]
        elif re.search('ReduceSum-op', k) is not None:
            assert v == [[dev_num, 1]]
