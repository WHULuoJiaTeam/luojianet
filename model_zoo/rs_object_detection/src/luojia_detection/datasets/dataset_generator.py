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

"""dataset"""
from __future__ import division

import os
import numpy as np
from numpy import random
import cv2
import luojianet_ms.dataset as de

from src.luojia_detection.configuration.config import config

import glob

if config.device_target == "Ascend":
    np_cast_type = np.float16
else:
    np_cast_type = np.float32


def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


class PhotoMetricDistortion:
    """Photo Metric Distortion"""
    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img, boxes, labels):
        # random brightness
        img = img.astype('float32')

        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        return img, boxes, labels


def rescale_with_tuple(img, scale):
    h, w = img.shape[:2]
    scale_factor = min(max(scale) / max(h, w), min(scale) / min(h, w))
    new_size = int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5)
    rescaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    return rescaled_img, scale_factor


def rescale_with_factor(img, scale_factor):
    h, w = img.shape[:2]
    new_size = int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5)
    return cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)


def rescale_column(img, img_shape, gt_bboxes, gt_label, gt_num, gt_mask):
    """rescale operation for image"""
    if config.enable_ms:
        idx = random.randint(0, len(config.multi_scales))
        img_w = config.multi_scales[idx][0]
        img_h = config.multi_scales[idx][1]
    else:
        img_w = config.img_width
        img_h = config.img_height
    img_data, scale_factor = rescale_with_tuple(img, (img_w, img_h))
    if img_data.shape[0] > img_h:
        img_data, scale_factor2 = rescale_with_tuple(img_data, (img_h, img_h))
        scale_factor = scale_factor * scale_factor2

    gt_bboxes = gt_bboxes * scale_factor
    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_data.shape[1] - 1)
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_data.shape[0] - 1)

    pad_h = config.img_height - img_data.shape[0]
    pad_w = config.img_width - img_data.shape[1]
    assert ((pad_h >= 0) and (pad_w >= 0))

    pad_img_data = np.zeros((config.img_height, config.img_width, 3)).astype(img_data.dtype)
    pad_img_data[0:img_data.shape[0], 0:img_data.shape[1], :] = img_data

    img_shape = (config.img_height, config.img_width, 1.0)
    img_shape = np.asarray(img_shape, dtype=np.float32)

    if config.mask_on:
        gt_mask_data = np.array([
            rescale_with_factor(mask, scale_factor)
            for mask in gt_mask
        ])
        mask_count, mask_h, mask_w = gt_mask_data.shape
        pad_mask = np.zeros((mask_count, config.img_height, config.img_width)).astype(gt_mask_data.dtype)
        pad_mask[:, 0:mask_h, 0:mask_w] = gt_mask_data

        return (pad_img_data, img_shape, gt_bboxes, gt_label, gt_num, pad_mask)
    else:
        return (pad_img_data, img_shape, gt_bboxes, gt_label, gt_num, gt_mask)


def rescale_column_test(img, img_shape, gt_bboxes, gt_label, gt_num, gt_mask):
    """rescale operation for image of eval"""
    img_data, scale_factor = rescale_with_tuple(img, (config.img_width, config.img_height))
    if img_data.shape[0] > config.img_height:
        img_data, scale_factor2 = rescale_with_tuple(img_data, (config.img_height, config.img_height))
        scale_factor = scale_factor*scale_factor2

    pad_h = config.img_height - img_data.shape[0]
    pad_w = config.img_width - img_data.shape[1]
    assert ((pad_h >= 0) and (pad_w >= 0))

    pad_img_data = np.zeros((config.img_height, config.img_width, 3)).astype(img_data.dtype)
    pad_img_data[0:img_data.shape[0], 0:img_data.shape[1], :] = img_data

    img_shape = np.append(img_shape, (scale_factor, scale_factor))
    img_shape = np.asarray(img_shape, dtype=np.float32)

    return (pad_img_data, img_shape, gt_bboxes, gt_label, gt_num, gt_mask)


def rescale_column_inference(img, img_shape, gt_bboxes, gt_label, gt_num, config):
    """rescale operation for image of eval"""
    img_data, scale_factor = rescale_with_tuple(img, (config.inference_img_width, config.inference_img_height))
    if img_data.shape[0] > config.img_height:
        img_data, scale_factor2 = rescale_with_tuple(img_data, (config.inference_img_height, config.inference_img_width))
        scale_factor = scale_factor*scale_factor2

    pad_h = config.inference_img_height - img_data.shape[0]
    pad_w = config.inference_img_width - img_data.shape[1]
    assert ((pad_h >= 0) and (pad_w >= 0))

    pad_img_data = np.zeros((config.inference_img_height, config.inference_img_width, 3)).astype(img_data.dtype)
    pad_img_data[0:img_data.shape[0], 0:img_data.shape[1], :] = img_data

    img_shape = np.append(img_shape, (scale_factor, scale_factor))
    img_shape = np.asarray(img_shape, dtype=np.float32)

    return (pad_img_data, img_shape, gt_bboxes, gt_label, gt_num)


def resize_column(img, img_shape, gt_bboxes, gt_label, gt_num, gt_mask):
    """resize operation for image"""
    img_data = img
    h, w = img_data.shape[:2]

    if config.enable_ms:
        idx = random.randint(0, len(config.multi_scales))
        img_w = config.multi_scales[idx][0]
        img_h = config.multi_scales[idx][1]
    else:
        img_w = config.img_width
        img_h = config.img_height

    img_data = cv2.resize(img_data, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    pad_h = config.img_height - img_h
    pad_w = config.img_width - img_w
    assert ((pad_h >= 0) and (pad_w >= 0))
    pad_img_data = np.zeros((config.img_height, config.img_width, 3)).astype(img_data.dtype)
    pad_img_data[0:img_data.shape[0], 0:img_data.shape[1], :] = img_data
    h_scale = img_h / h
    w_scale = img_w / w

    scale_factor = np.array(
        [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    img_shape = (config.img_height, config.img_width, 1.0)
    img_shape = np.asarray(img_shape, dtype=np.float32)

    gt_bboxes = gt_bboxes * scale_factor
    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1) # x1, x2   [0, W-1]
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1) # y1, y2   [0, H-1]

    if config.mask_on:
        gt_mask_data = np.array([
            cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            for mask in gt_mask
        ])
        mask_count, mask_h, mask_w = gt_mask_data.shape
        pad_mask = np.zeros((mask_count, config.img_height, config.img_width)).astype(gt_mask_data.dtype)
        pad_mask[:, 0:mask_h, 0:mask_w] = gt_mask_data

        return (pad_img_data, img_shape, gt_bboxes, gt_label, gt_num, pad_mask)
    else:
        return (pad_img_data, img_shape, gt_bboxes, gt_label, gt_num, gt_mask)


def resize_column_test(img, img_shape, gt_bboxes, gt_label, gt_num, gt_mask):
    """resize operation for image of eval"""
    img_data = img
    h, w = img_data.shape[:2]
    img_data = cv2.resize(img_data, (config.img_width, config.img_height), interpolation=cv2.INTER_LINEAR)
    h_scale = config.img_height / h
    w_scale = config.img_width / w

    scale_factor = np.array(
        [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    img_shape = np.append(img_shape, (h_scale, w_scale))
    img_shape = np.asarray(img_shape, dtype=np.float32)

    gt_bboxes = gt_bboxes * scale_factor

    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)

    return (img_data, img_shape, gt_bboxes, gt_label, gt_num, gt_mask)


def resize_column_inference(img, img_shape, gt_bboxes, gt_label, gt_num, config):
    """resize operation for image of eval"""
    img_data = img
    h, w = img_data.shape[:2]
    img_data = cv2.resize(
        img_data, (config.inference_img_width, config.inference_img_height), interpolation=cv2.INTER_LINEAR)
    h_scale = config.inference_img_height / h
    w_scale = config.inference_img_width / w

    scale_factor = np.array(
        [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    img_shape = np.append(img_shape, (h_scale, w_scale))
    img_shape = np.asarray(img_shape, dtype=np.float32)

    gt_bboxes = gt_bboxes * scale_factor

    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)

    return (img_data, img_shape, gt_bboxes, gt_label, gt_num)


def imnormalize_column(img, img_shape, gt_bboxes, gt_label, gt_num, gt_mask):
    """imnormalize operation for image"""
    mean = np.asarray([123.675, 116.28, 103.53])
    std = np.asarray([58.395, 57.12, 57.375])
    img_data = img.copy().astype(np.float32)
    cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB, img_data)  # inplace
    cv2.subtract(img_data, np.float64(mean.reshape(1, -1)), img_data)  # inplace
    cv2.multiply(img_data, 1 / np.float64(std.reshape(1, -1)), img_data)  # inplace

    img_data = img_data.astype(np.float32)

    return (img_data, img_shape, gt_bboxes, gt_label, gt_num, gt_mask)


def imnormalize_column_inference(img, img_shape, gt_bboxes, gt_label, gt_num):
    """imnormalize operation for image"""
    mean = np.asarray([123.675, 116.28, 103.53])
    std = np.asarray([58.395, 57.12, 57.375])
    img_data = img.copy().astype(np.float32)
    cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB, img_data)  # inplace
    cv2.subtract(img_data, np.float64(mean.reshape(1, -1)), img_data)  # inplace
    cv2.multiply(img_data, 1 / np.float64(std.reshape(1, -1)), img_data)  # inplace

    img_data = img_data.astype(np.float32)
    return (img_data, img_shape, gt_bboxes, gt_label, gt_num)


def flip_column(img, img_shape, gt_bboxes, gt_label, gt_num, gt_mask):
    """flip operation for image"""
    img_data = img
    img_data = np.flip(img_data, axis=1)
    flipped = gt_bboxes.copy()
    _, w, _ = img_data.shape

    flipped[..., 0::4] = w - gt_bboxes[..., 2::4] - 1  # x1 = W-x2-1
    flipped[..., 2::4] = w - gt_bboxes[..., 0::4] - 1  # x2 = W-x1-1

    if config.mask_on:
        gt_mask_data = np.array([mask[:, ::-1] for mask in gt_mask])
        return (img_data, img_shape, flipped, gt_label, gt_num, gt_mask_data)
    else:
        return (img_data, img_shape, flipped, gt_label, gt_num, gt_mask)


def transpose_column(img, img_shape, gt_bboxes, gt_label, gt_num, gt_mask):
    """transpose operation for image"""
    img_data = img.transpose(2, 0, 1).copy()
    img_data = img_data.astype(np_cast_type)
    img_shape = img_shape.astype(np_cast_type)
    gt_bboxes = gt_bboxes.astype(np_cast_type)
    gt_label = gt_label.astype(np.int32)
    gt_num = gt_num.astype(np.bool_)

    if config.mask_on:
        gt_mask_data = gt_mask.astype(np.bool_)
        return (img_data, img_shape, gt_bboxes, gt_label, gt_num, gt_mask_data)
    else:
        return (img_data, img_shape, gt_bboxes, gt_label, gt_num)


def transpose_column_inference(img, img_shape, gt_bboxes, gt_label, gt_num):
    """transpose operation for image"""
    img_data = img.transpose(2, 0, 1).copy()
    img_data = img_data.astype(np.float32)
    img_shape = img_shape.astype(np.float32)
    return (img_data, img_shape)


def photo_crop_column(img, img_shape, gt_bboxes, gt_label, gt_num, gt_mask):
    """photo crop operation for image"""
    random_photo = PhotoMetricDistortion()
    img_data, gt_bboxes, gt_label = random_photo(img, gt_bboxes, gt_label)

    return (img_data, img_shape, gt_bboxes, gt_label, gt_num, gt_mask)


def pad_to_max(img, img_shape, gt_bboxes, gt_label, gt_num, gt_mask, instance_count):
    pad_max_number = config.max_instance_count
    gt_box_new = np.pad(gt_bboxes, ((0, pad_max_number - instance_count), (0, 0)), mode="constant", constant_values=0)
    gt_label_new = np.pad(gt_label, ((0, pad_max_number - instance_count)), mode="constant", constant_values=-1)
    gt_iscrowd_new = np.pad(gt_num, ((0, pad_max_number - instance_count)), mode="constant", constant_values=1)
    gt_iscrowd_new_revert = ~(gt_iscrowd_new.astype(np.bool_))

    if config.mask_on:
        gt_mask_new = np.pad(np.array(gt_mask).astype(int), ((0, pad_max_number - instance_count), (0, 0), (0, 0)), mode="constant", constant_values=0)
        return img, img_shape, gt_box_new, gt_label_new, gt_iscrowd_new_revert, gt_mask_new
    else:
        return img, img_shape, gt_box_new, gt_label_new, gt_iscrowd_new_revert, gt_mask


def preprocess_fn(image, box, mask_info=None, img_shape_info=None, is_training=False):
    """Preprocess function for dataset."""
    def _infer_data(image_bgr, image_shape, gt_box_new, gt_label_new, gt_iscrowd_new_revert,
                    gt_mask_new, instance_count):
        image_shape = image_shape[:2]
        input_data = image_bgr, image_shape, gt_box_new, gt_label_new, gt_iscrowd_new_revert, gt_mask_new

        if config.keep_ratio:
            input_data = rescale_column_test(*input_data)
        else:
            input_data = resize_column_test(*input_data)
        input_data = imnormalize_column(*input_data)

        input_data = pad_to_max(*input_data, instance_count)
        output_data = transpose_column(*input_data)
        return output_data

    def _data_aug(image, box, mask_info, img_shape_info, is_training):
        """Data augmentation function."""
        image_bgr = image.copy()
        image_bgr[:, :, 0] = image[:, :, 2]
        image_bgr[:, :, 1] = image[:, :, 1]
        image_bgr[:, :, 2] = image[:, :, 0]
        image_shape = image_bgr.shape[:2]
        instance_count = box.shape[0]
        gt_box = box[:, :4]
        gt_label = box[:, 4]
        gt_iscrowd = box[:, 5]

        if config.mask_on:
            instance_masks = []
            if len(mask_info) != 0:
                for id, segm in enumerate(mask_info):
                    m = annToMask(segm.tolist(), img_shape_info[0], img_shape_info[1])
                    if m.max() < 1:
                        print("all black mask!!!!")
                        continue
                    if gt_iscrowd[id] and (m.shape[0] != img_shape_info[0] or m.shape[1] != img_shape_info[1]):
                        m = np.ones([img_shape_info[0], img_shape_info[1]], dtype=np.int8)
                    instance_masks.append(m)
                instance_masks = np.stack(instance_masks, axis=0).astype(np.int8)
                gt_mask = np.array(instance_masks)
                mask_shape = np.array(instance_masks.shape, dtype=np.int32)
            else:
                gt_mask = np.zeros([1, img_shape_info[0], img_shape_info[1]], dtype=np.int8)
                mask_shape = np.array([1, img_shape_info[0], img_shape_info[1]], dtype=np.int32)

            n, h, w = mask_shape
            gt_mask = gt_mask.reshape(n, h, w)
            assert n == box.shape[0]
        else:
            gt_mask = None

        if not is_training:
            return _infer_data(image_bgr, image_shape, gt_box, gt_label, gt_iscrowd, gt_mask, instance_count)

        flip = (np.random.rand() < config.flip_ratio)
        input_data = image_bgr, image_shape, gt_box, gt_label, gt_iscrowd, gt_mask

        if config.keep_ratio:
            input_data = rescale_column(*input_data)
        else:
            input_data = resize_column(*input_data)
        if flip:
            input_data = flip_column(*input_data)

        input_data = imnormalize_column(*input_data)

        input_data = pad_to_max(*input_data, instance_count)
        output_data = transpose_column(*input_data)
        return output_data

    return _data_aug(image, box, mask_info, img_shape_info, is_training)


def preprocess_fn_inference(image, img_path, config):
    """Preprocess function for dataset."""
    def _infer_data(image_bgr, image_shape):
        image_shape = image_shape[:2]
        image_shape = np.append(image_shape, (1.0, 1.0))
        input_data = image_bgr, image_shape, None, None, None

        input_data = imnormalize_column_inference(*input_data)

        output_data = transpose_column_inference(*input_data)
        return output_data

    def _data_aug(image):
        """Data augmentation function."""
        image_bgr = image.copy()
        image_bgr[:, :, 0] = image[:, :, 2]
        image_bgr[:, :, 1] = image[:, :, 1]
        image_bgr[:, :, 2] = image[:, :, 0]
        image_shape = image_bgr.shape[:2]

        return _infer_data(image_bgr, image_shape)

    return _data_aug(image)


def annToMask(segm, height, width):
    """Convert annotation to RLE and then to binary mask."""
    from pycocotools import mask as maskHelper
    if isinstance(segm, list):
        rles = maskHelper.frPyObjects(segm, height, width)
        rle = maskHelper.merge(rles)
    elif isinstance(segm['counts'], list):
        rle = maskHelper.frPyObjects(segm, height, width)
    else:
        rle = segm
    m = maskHelper.decode(rle)
    return m


def create_coco_label(is_training):
    """Get image path and annotation from COCO."""
    from pycocotools.coco import COCO
    coco_root = config.coco_root
    data_type = config.val_data_type
    if is_training:
        data_type = config.train_data_type
    # Classes need to train or test.
    train_cls = config.coco_classes

    train_cls_dict = {}
    for i, cls in enumerate(train_cls):
        train_cls_dict[cls] = i

    anno_json = os.path.join(coco_root, config.instance_set.format(data_type))

    coco = COCO(anno_json)
    classs_dict = {}
    cat_ids = coco.loadCats(coco.getCatIds())
    for cat in cat_ids:
        classs_dict[cat["id"]] = cat["name"]

    image_ids = coco.getImgIds()
    image_files = []
    image_anno_dict = {}
    images_num = len(image_ids)

    if config.mask_on:
        masks_info = {}
        imgs_shape_info = {}
    for ind, img_id in enumerate(image_ids):
        image_info = coco.loadImgs(img_id)
        file_name = image_info[0]["file_name"]
        image_path = os.path.join(coco_root, data_type, file_name)
        if not os.path.isfile(image_path):
            print("{}/{}: {} is in annotations but not exist".format(ind + 1, images_num, image_path))
            continue
        anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = coco.loadAnns(anno_ids)
        image_path = os.path.join(coco_root, data_type, file_name)
        annos = []
        if (ind + 1) % 10 == 0:
            print("{}/{}: parsing annotation for image={}".format(ind + 1, images_num, file_name))

        if config.mask_on:
            instance_masks = []
            image_height = coco.imgs[img_id]["height"]
            image_width = coco.imgs[img_id]["width"]

        for label in anno:
            bbox = label["bbox"]
            class_name = classs_dict[label["category_id"]]
            if class_name in train_cls:
                if config.mask_on:
                    instance_masks.append(label["segmentation"])
                # get coco bbox
                x1, x2 = bbox[0], bbox[0] + bbox[2]
                y1, y2 = bbox[1], bbox[1] + bbox[3]
                annos.append([x1, y1, x2, y2] + [train_cls_dict[class_name]] + [int(label["iscrowd"])])
            else:
                print("not in classes: ", class_name)

        image_files.append(image_path)
        if annos:
            image_anno_dict[image_path] = np.array(annos)
            if config.mask_on:
                masks_info[image_path] = instance_masks
                imgs_shape_info[image_path] = [image_height, image_width]

        else:
            print("no annotations for image ", file_name)
            image_anno_dict[image_path] = np.array([0, 0, 0, 0, 0, 1])
            if config.mask_on:
                masks_info[image_path] = []
                imgs_shape_info[image_path] = [image_height, image_width]
    if config.mask_on:
        return image_files, image_anno_dict, masks_info, imgs_shape_info
    else:
        return image_files, image_anno_dict


def create_maskrcnn_dataset(batch_size=1, device_num=1, rank_id=0, is_training=True,
                              num_parallel_workers=2):
    """Create FasterRcnn dataset with MindDataset."""
    cv2.setNumThreads(0)

    ds_generator = DatasetGenerator(is_training)
    ds = de.GeneratorDataset(ds_generator, column_names=["image", "annotation", "mask_info", "img_shape_info"],
                             num_shards=device_num, shard_id=rank_id, num_parallel_workers=1, shuffle=is_training)
    compose_map_func = (lambda image, annotation, mask_info, img_shape_info:
                        preprocess_fn(image, annotation, mask_info, img_shape_info, is_training))

    if is_training:
        ds = ds.map(input_columns=["image", "annotation", "mask_info", "img_shape_info"],
                    output_columns=["image", "image_shape", "box", "label", "valid_num", "mask"],
                    column_order=["image", "image_shape", "box", "label", "valid_num", "mask"],
                    operations=compose_map_func, python_multiprocessing=False,
                    num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=True)
    else:
        ds = ds.map(input_columns=["image", "annotation", "mask_info", "img_shape_info"],
                    output_columns=["image", "image_shape", "box", "label", "valid_num", "mask"],
                    column_order=["image", "image_shape", "box", "label", "valid_num", "mask"],
                    operations=compose_map_func,
                    num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=True)
    return ds


def create_fasterrcnn_dataset(batch_size=1, device_num=1, rank_id=0, is_training=True,
                              num_parallel_workers=2):
    """Create FasterRcnn dataset with MindDataset."""
    cv2.setNumThreads(0)

    ds_generator = DatasetGenerator(is_training)
    ds = de.GeneratorDataset(ds_generator, column_names=["image", "annotation"],
                             num_shards=device_num, shard_id=rank_id, num_parallel_workers=1, shuffle=is_training)
    compose_map_func = (lambda image, annotation:
                        preprocess_fn(image, annotation, is_training=is_training))

    if is_training:
        ds = ds.map(input_columns=["image", "annotation"],
                    output_columns=["image", "image_shape", "box", "label", "valid_num"],
                    column_order=["image", "image_shape", "box", "label", "valid_num"],
                    operations=compose_map_func, python_multiprocessing=False,
                    num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=True)
    else:
        ds = ds.map(input_columns=["image", "annotation"],
                    output_columns=["image", "image_shape", "box", "label", "valid_num"],
                    column_order=["image", "image_shape", "box", "label", "valid_num"],
                    operations=compose_map_func,
                    num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=True)
    return ds


def create_dataset_inference(device_num=1, rank_id=0, num_parallel_workers=2):
    """Create FasterRcnn dataset with MindDataset."""
    ds_generator = DatasetGenerator_inference()
    img_paths = ds_generator.img_files
    dataset = de.GeneratorDataset(ds_generator, column_names=["image", "img_paths"],
                             num_shards=device_num, shard_id=rank_id, num_parallel_workers=1, shuffle=False)
    compose_map_func = (lambda image, img_path: preprocess_fn_inference(image, img_path, config=config))

    dataset = dataset.map(input_columns=["image", "img_paths"],
                          output_columns=["image", "image_shape"],
                          column_order=["image", "image_shape"],
                          operations=compose_map_func, python_multiprocessing=False,
                          num_parallel_workers=num_parallel_workers)
    dataset = dataset.batch(batch_size=1, drop_remainder=True)
    return dataset, img_paths


class DatasetGenerator(object):
    def __init__(self, is_training=True):
        self.is_training = is_training
        if config.mask_on:
            self.img_files, self.img_anno_dict, self.masks_info, self.imgs_shape_info = create_coco_label(is_training)
        else:
            self.img_files, self.img_anno_dict = create_coco_label(is_training)

    def __getitem__(self, idx):
        # image
        img_path = self.img_files[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # label
        img_anno = self.img_anno_dict[img_path]
        if config.mask_on:
            mask_info = self.masks_info[img_path]
            img_shape_info = self.imgs_shape_info[img_path]
            return img, img_anno, mask_info, img_shape_info
        else:
            return img, img_anno

    def __len__(self):
        return len(self.img_files)


class DatasetGenerator_inference(object):
    def __init__(self):
        self.img_files = glob.glob(config.inference_img_dir + '*.png')

    def __getitem__(self, idx):
        # image
        img_path = self.img_files[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, img_path

    def __len__(self):
        return len(self.img_files)


def get_mask_box(mask, img_h, img_w, x_append=0, y_append=0):
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 1:
        contour_all = contours[0][0]
    else:
        contour_all = contours[0][0]
        for i in range(len(contours[0]) - 1):
            contour_all = np.concatenate((contour_all, contours[0][i + 1]), axis=0)
    rect = cv2.minAreaRect(contour_all)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    for i in range(4):
        box[i][0] = box[i][0] + x_append
        box[i][1] = box[i][1] + y_append
        box[i][0] = max(box[i][0], 0)
        box[i][0] = min(box[i][0], img_w)
        box[i][1] = max(box[i][1], 0)
        box[i][1] = min(box[i][1], img_h)

    return box


if __name__ == "__main__":
    from PIL import Image

    print(np.__version__)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    dataset = create_fasterrcnn_dataset(batch_size=config.batch_size, device_num=1, rank_id=0, is_training=True)

    cnt = 0
    for idx, data in enumerate(dataset.create_dict_iterator()):
        img_data = data['image']
        img_metas = data['image_shape']
        gt_bboxes = data['box']
        gt_labels = data['label']
        gt_num = data['valid_num']
        # gt_mask = data["mask"]

        # for img, bboxes, maskes, num in zip(img_data,gt_bboxes, gt_mask, gt_num):
        for img, bboxes, num in zip(img_data, gt_bboxes, gt_num):
            mean = np.asarray([123.675, 116.28, 103.53])
            std = np.asarray([58.395, 57.12, 57.375])
            img = img.asnumpy().astype(np.float32)
            img = np.squeeze(img).transpose(1, 2, 0)
            img = np.multiply(img, std)
            img = np.add(img, mean + 1)
            img = img.astype(np.uint8)

            img = Image.fromarray(img)
            img.save("/dat02/hhb/luojiaNet/src/luojia_detection/datasets/test.png")

            img = cv2.imread("/dat02/hhb/luojiaNet/src/luojia_detection/datasets/test.png")
            # for bbox, mask, is_ins in zip(bboxes, maskes, num):
            for bbox, is_ins in zip(bboxes, num):
                if is_ins:
                    bbox = bbox.asnumpy()
                    # mask = mask.asnumpy().astype(np.uint8)
                    # cur_rbbox = get_mask_box(mask, 1536, 1536, 0, 0).reshape(4, 2)
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255),
                                  thickness=3)
                    # cv2.polylines(img, [cur_rbbox], 1, (0, 255, 0), thickness=3)
                else:
                    break
            cv2.imwrite("/dat02/hhb/luojiaNet/src/luojia_detection/datasets/test_" + str(cnt) +".png", img)
            cnt += 1
            pass
