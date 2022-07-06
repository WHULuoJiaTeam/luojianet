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
"""MaskRcnn Rcnn for mask network."""

import numpy as np
import luojianet_ms.common.dtype as mstype
import luojianet_ms.nn as nn
from luojianet_ms.ops import operations as P
from luojianet_ms.common.tensor import Tensor
from luojianet_ms import context


def _conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0, pad_mode='pad', gain=1):
    """Conv2D wrapper."""
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    # xavier_normal
    fan_in = in_channels * kernel_size * kernel_size
    fan_out = out_channels * kernel_size * kernel_size
    std = gain * (2 / (fan_in + fan_out)) ** 0.5
    weights = Tensor(np.random.normal(loc=0.0, scale=std, size=shape).astype(np.float32))
    shape_bias = (out_channels,)
    bias = Tensor(np.array(np.zeros(shape_bias)).astype(np.float32))
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     pad_mode=pad_mode, weight_init=weights, has_bias=True, bias_init=bias)


def _convTanspose(in_channels, out_channels, kernel_size=1, stride=1, padding=0, pad_mode='pad',
                  gain=1):
    """ConvTranspose wrapper."""
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    # xavier_normal
    fan_in = in_channels * kernel_size * kernel_size
    fan_out = out_channels * kernel_size * kernel_size
    std = gain * (2 / (fan_in + fan_out)) ** 0.5
    weights = Tensor(np.random.normal(loc=0.0, scale=std, size=shape).astype(np.float32))
    shape_bias = (out_channels,)
    bias = Tensor(np.array(np.zeros(shape_bias)).astype(np.float32))
    return nn.Conv2dTranspose(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              pad_mode=pad_mode, weight_init=weights, has_bias=True, bias_init=bias)


class FpnMask(nn.Module):
    """conv layers of mask head"""

    def __init__(self, input_channels, output_channels, num_classes):
        super(FpnMask, self).__init__()

        if context.get_context("device_target") == "Ascend":
            self.cast_type = mstype.float16
        else:
            self.cast_type = mstype.float32

        self.mask_conv1 = _conv(input_channels, output_channels, kernel_size=3, gain=2 ** 0.5,
                                pad_mode="same").to_float(self.cast_type)
        self.mask_relu1 = P.ReLU()

        self.mask_conv2 = _conv(output_channels, output_channels, kernel_size=3, gain=2 ** 0.5,
                                pad_mode="same").to_float(self.cast_type)
        self.mask_relu2 = P.ReLU()

        self.mask_conv3 = _conv(output_channels, output_channels, kernel_size=3, gain=2 ** 0.5,
                                pad_mode="same").to_float(self.cast_type)
        self.mask_relu3 = P.ReLU()

        self.mask_conv4 = _conv(output_channels, output_channels, kernel_size=3, gain=2 ** 0.5,
                                pad_mode="same").to_float(self.cast_type)
        self.mask_relu4 = P.ReLU()

        self.mask_deconv5 = _convTanspose(output_channels, output_channels, kernel_size=2, gain=2 ** 0.5,
                                          stride=2, pad_mode="valid").to_float(self.cast_type)
        self.mask_relu5 = P.ReLU()
        self.mask_conv6 = _conv(output_channels, num_classes, kernel_size=1, stride=1, gain=2,
                                pad_mode="valid").to_float(self.cast_type)

    def forward(self, x):
        x = self.mask_conv1(x)
        x = self.mask_relu1(x)

        x = self.mask_conv2(x)
        x = self.mask_relu2(x)

        x = self.mask_conv3(x)
        x = self.mask_relu3(x)

        x = self.mask_conv4(x)
        x = self.mask_relu4(x)

        x = self.mask_deconv5(x)
        x = self.mask_relu5(x)

        x = self.mask_conv6(x)

        return x


class RcnnMask(nn.Module):
    """
    Rcnn for mask subnet.

    Args:
        config (dict) - Config.
        batch_size (int) - Batchsize.
        num_classes (int) - Class number.
        target_means (list) - Means for encode function. Default: (.0, .0, .0, .0]).
        target_stds (list) - Stds for encode function. Default: (0.1, 0.1, 0.2, 0.2).

    Returns:
        Tuple, tuple of output tensor.

    Examples:
        RcnnMask(config=config, representation_size = 1024, batch_size=2, num_classes = 81, \
             target_means=(0., 0., 0., 0.), target_stds=(0.1, 0.1, 0.2, 0.2))
    """

    def __init__(self,
                 config,
                 batch_size,
                 num_classes,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2)
                 ):
        super(RcnnMask, self).__init__()
        cfg = config

        if context.get_context("device_target") == "Ascend":
            self.cast_type = mstype.float16
            self.np_cast_type = np.float16
        else:
            self.cast_type = mstype.float32
            self.np_cast_type = np.float32

        self.rcnn_loss_mask_fb_weight = Tensor(np.array(cfg.rcnn_loss_mask_fb_weight).astype(self.np_cast_type))
        self.rcnn_mask_out_channels = cfg.rcnn_mask_out_channels
        self.target_means = target_means
        self.target_stds = target_stds
        self.num_classes = num_classes
        self.in_channels = cfg.rcnn_in_channels

        self.fpn_mask = FpnMask(self.in_channels, self.rcnn_mask_out_channels, self.num_classes)

        self.logicaland = P.LogicalAnd()
        self.loss_mask = P.SigmoidCrossEntropyWithLogits()
        self.onehot = P.OneHot()
        self.greater = P.Greater()
        self.cast = P.Cast()
        self.sum_loss = P.ReduceSum()
        self.tile = P.Tile()
        self.expandims = P.ExpandDims()

        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)

        self.num_bboxes = cfg.num_expected_pos_stage2 * batch_size
        rmv_first = np.ones((self.num_bboxes, self.num_classes))
        rmv_first[:, 0] = np.zeros((self.num_bboxes,))
        self.rmv_first_tensor = Tensor(rmv_first.astype(self.np_cast_type))
        self.mean_loss = P.ReduceMean()

    def forward(self, mask_featuremap, labels=None, mask=None, mask_fb_targets=None):
        x_mask_fb = self.fpn_mask(mask_featuremap)

        if self.training:
            bbox_weights = self.cast(self.logicaland(self.greater(labels, 0), mask), mstype.int32) * labels
            mask_fb_targets = self.tile(self.expandims(mask_fb_targets, 1), (1, self.num_classes, 1, 1))

            loss_mask_fb = self.loss(x_mask_fb, bbox_weights, mask, mask_fb_targets)
            out = loss_mask_fb
        else:
            out = x_mask_fb

        return out

    def loss(self, masks_fb_pred, bbox_weights, weights, masks_fb_targets):
        """Loss method."""
        weights = self.cast(weights, self.cast_type)
        bbox_weights = self.cast(self.onehot(bbox_weights, self.num_classes, self.on_value, self.off_value),
                                 self.cast_type)
        bbox_weights = bbox_weights * self.rmv_first_tensor  # * self.rmv_first_tensor  exclude background

        # loss_mask_fb
        masks_fb_targets = self.cast(masks_fb_targets, self.cast_type)
        loss_mask_fb = self.loss_mask(masks_fb_pred, masks_fb_targets)
        loss_mask_fb = self.mean_loss(loss_mask_fb, (2, 3))
        loss_mask_fb = loss_mask_fb * bbox_weights
        loss_mask_fb = loss_mask_fb / (self.sum_loss(weights, (0,)) + 1e-5)
        loss_mask_fb = self.sum_loss(loss_mask_fb, (0, 1))

        return loss_mask_fb
