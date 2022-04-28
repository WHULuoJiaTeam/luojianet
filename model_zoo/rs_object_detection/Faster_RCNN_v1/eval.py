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

"""Evaluation for FasterRcnn"""
import os
import time
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
import luojianet_ms.common.dtype as mstype
from luojianet_ms import context
from luojianet_ms.train.serialization import load_checkpoint, load_param_into_net
from luojianet_ms.common import set_seed, Parameter

from src.dataset import data_to_mindrecord_byte_image, create_fasterrcnn_dataset, parse_json_annos_from_txt
from src.util import coco_eval, bbox2result_1image, results2json
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.FasterRcnn.faster_rcnn import Faster_Rcnn

from src.dataset_generator import create_my_fasterrcnn_dataset


def fasterrcnn_eval(ckpt_path, anno_path):
    """FasterRcnn evaluation."""
    if not os.path.isfile(ckpt_path):
        raise RuntimeError("CheckPoint file {} is not valid.".format(ckpt_path))

    ds = create_my_fasterrcnn_dataset(config, batch_size=1, is_training=False)
    net = Faster_Rcnn(config)

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

    if config.dataset != "coco":
        dataset_coco = COCO()
        dataset_coco.dataset, dataset_coco.anns, dataset_coco.cats, dataset_coco.imgs = dict(), dict(), dict(), dict()
        dataset_coco.imgToAnns, dataset_coco.catToImgs = defaultdict(list), defaultdict(list)
        dataset_coco.dataset = parse_json_annos_from_txt(anno_path, config)
        dataset_coco.createIndex()
    else:
        dataset_coco = COCO(anno_path)

    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    max_num = 128
    for idx, data in enumerate(ds.create_dict_iterator(num_epochs=1)):
        eval_iter = eval_iter + 1
        img_data = data['image']
        img_metas = data['image_shape']
        gt_bboxes = data['box']  # not effect, so it is the original number
        gt_labels = data['label']
        gt_num = data['valid_num']

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

        for j in range(config.test_batch_size):
            all_bbox_squee = np.squeeze(all_bbox.asnumpy()[j, :, :])
            all_label_squee = np.squeeze(all_label.asnumpy()[j, :, :])
            all_mask_squee = np.squeeze(all_mask.asnumpy()[j, :, :])

            all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
            all_labels_tmp_mask = all_label_squee[all_mask_squee]

            if all_bboxes_tmp_mask.shape[0] > max_num:
                inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
                inds = inds[:max_num]
                all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
                all_labels_tmp_mask = all_labels_tmp_mask[inds]

            outputs_tmp = bbox2result_1image(all_bboxes_tmp_mask, all_labels_tmp_mask, config.num_classes)

            outputs.append(outputs_tmp)

    eval_types = ["bbox"]
    result_files = results2json(dataset_coco, outputs, "./results.pkl")

    coco_eval(config, result_files, eval_types, dataset_coco, single_result=True, plot_detect_result=True)


def modelarts_pre_process():
    pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def eval_fasterrcnn():
    """ eval_fasterrcnn """
    print("Start Eval!")
    fasterrcnn_eval(config.eval_checkpoint_path, config.eval_anno_path)  # change config-param names

    flags = [0] * 3
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
    import os
    # set visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device_id

    # set random seed
    set_seed(1)
    # context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=int(config.device_id))
    context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target, device_id=int(config.device_id))

    # run eval
    eval_fasterrcnn()
