# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""Evaluation for MaskRcnn"""
import os
import time
import numpy as np

from src.luojia_detection.configuration.config import config
from src.luojia_detection.env.moxing_adapter import moxing_wrapper
from src.luojia_detection.env.device_adapter import get_device_id, get_device_num
from src.luojia_detection.detectors import Mask_Rcnn_Resnet,Faster_Rcnn_Resnet
from src.luojia_detection.datasets import create_maskrcnn_dataset, create_fasterrcnn_dataset
from src.luojia_detection.utils import coco_eval_maskrcnn, coco_eval_fasterrcnn, bbox2result_1image, results2json, get_seg_masks

from pycocotools.coco import COCO
from luojianet_ms import context, Tensor
from luojianet_ms.train.serialization import load_checkpoint, load_param_into_net
from luojianet_ms.common import set_seed, Parameter
import luojianet_ms.common.dtype as mstype


def eval(ckpt_path, ann_file):
    """evaluation."""
    if config.mask_on:
        ds = create_maskrcnn_dataset(batch_size=config.test_batch_size, is_training=False)
        net = Mask_Rcnn_Resnet(config)
    else:
        ds = create_fasterrcnn_dataset(batch_size=config.test_batch_size, is_training=False)
        net = Faster_Rcnn_Resnet(config)

    param_dict = load_checkpoint(ckpt_path)
    if config.device_target == "GPU":
        for key, value in param_dict.items():
            tensor = value.asnumpy().astype(np.float32)
            param_dict[key] = Parameter(tensor, key)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    device_type = "Ascend" if context.get_context("device_target") == "Ascend" else "Others"
    if device_type == "Ascend":
        net.to_float(mstype.float16)

    eval_iter = 0
    total = ds.get_dataset_size()
    outputs = []

    dataset_coco = COCO(ann_file)

    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    max_num = 128
    for data in ds.create_dict_iterator(num_epochs=1):
        eval_iter = eval_iter + 1

        img_data = data['image']
        img_metas = data['image_shape']
        gt_bboxes = data['box']
        gt_labels = data['label']
        gt_num = data['valid_num']

        if config.mask_on:
            gt_mask = data["mask"]
            start = time.time()
            # run net
            output = net(img_data, img_metas, gt_bboxes, gt_labels, gt_num, gt_mask)
            end = time.time()
            print("Iter {} cost time {}".format(eval_iter, end - start))
        else:
            start = time.time()
            # run net
            output = net(img_data, img_metas, gt_bboxes, gt_labels, gt_num)
            # output = net(*data)
            end = time.time()
            print("Iter {} cost time {}".format(eval_iter, end - start))

        # output
        all_bbox = output[0]
        all_label = output[1]
        all_mask = output[2]
        if config.mask_on:
            all_mask_fb = output[3]

        for j in range(config.test_batch_size):
            all_bbox_squee = np.squeeze(all_bbox.asnumpy()[j, :, :])
            all_label_squee = np.squeeze(all_label.asnumpy()[j, :, :])
            all_mask_squee = np.squeeze(all_mask.asnumpy()[j, :, :])
            all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
            all_labels_tmp_mask = all_label_squee[all_mask_squee]

            if config.mask_on:
                all_mask_fb_squee = np.squeeze(all_mask_fb.asnumpy()[j, :, :, :])
                all_mask_fb_tmp_mask = all_mask_fb_squee[all_mask_squee, :, :]

            if all_bboxes_tmp_mask.shape[0] > max_num:
                inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
                inds = inds[:max_num]
                all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
                all_labels_tmp_mask = all_labels_tmp_mask[inds]
                if config.mask_on:
                    all_mask_fb_tmp_mask = all_mask_fb_tmp_mask[inds]

            bbox_results = bbox2result_1image(all_bboxes_tmp_mask, all_labels_tmp_mask, config.num_classes)

            if config.mask_on:
                segm_results = get_seg_masks(all_mask_fb_tmp_mask, all_bboxes_tmp_mask, all_labels_tmp_mask, img_metas[j],
                                             True, config.num_classes)
                outputs.append((bbox_results, segm_results))
            else:
                outputs.append(bbox_results)

    if config.mask_on:
        eval_types = ["bbox", "segm"]
        result_files = results2json(dataset_coco, outputs, config.eval_save_dir + "results.pkl")
        coco_eval_maskrcnn(result_files, eval_types, dataset_coco, single_result=True)
    else:
        eval_types = ["bbox"]
        result_files = results2json(dataset_coco, outputs, config.eval_save_dir + "results.pkl")
        coco_eval_fasterrcnn(config, result_files, eval_types, dataset_coco, single_result=True, plot_detect_result=True)


def modelarts_process():
    pass


@moxing_wrapper(pre_process=modelarts_process)
def eval_():
    print('\neval.py config:\n', config)
    if not os.path.exists(config.eval_save_dir):
        os.mkdir(config.eval_save_dir)

    print("Start Eval!")
    eval(config.eval_checkpoint_path, config.ann_file)

    if not config.mask_on:
        flags = [0] * 3
        # config.eval_result_path = os.path.abspath("./eval_result")
        config.eval_result_path = os.path.abspath(config.eval_save_dir)
        if os.path.exists(config.eval_result_path):
            result_files = os.listdir(config.eval_result_path)
            for file in result_files:
                if file == "statistics.csv":
                    with open(os.path.join(config.eval_result_path, "statistics.csv"), "r") as f:
                        res = f.readlines()
                    if len(res) > 1:
                        if "class_name" in res[3] and "tp_num" in res[3] and len(res[4].strip().split(",")) > 1:
                            flags[0] = 1
                elif file in ("precision_ng_images", "recall_ng_images", "ok_images"):
                    imgs = os.listdir(os.path.join(config.eval_result_path, file))
                    if imgs:
                        flags[1] = 1

                elif file == "pr_curve_image":
                    imgs = os.listdir(os.path.join(config.eval_result_path, "pr_curve_image"))
                    if imgs:
                        flags[2] = 1
                else:
                    pass

        if sum(flags) == 3:
            print("eval success.")
            exit(0)
        else:
            print("eval failed.")
            exit(-1)


if __name__ == '__main__':
    """
    python eval.py --enable_eval \
                   --eval_dataset=./examples/mini_dataset \
                   --annotation=./examples/mini_dataset/train.json \
                   --result_save_path=./output_dir/eval_result/ \
                   --checkpoint_path=./pretrained_models/faster_rcnn-30_2253.ckpt
                   
    python eval.py --enable_eval \
                   --eval_dataset=./examples/mini_dataset \
                   --annotation=./examples/mini_dataset/train.json \
                   --result_save_path=./output_dir/eval_result/ \
                   --checkpoint_path=./pretrained_models/mask_rcnn-30_2253.ckpt
    """
    # set random seed
    set_seed(1)

    # set environment
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=config.device_id)

    # run evaluation
    eval_()
