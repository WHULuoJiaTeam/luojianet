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

"""Evaluation for FasterRcnn"""
import os
import time
import numpy as np

import luojianet_ms.common.dtype as mstype
from luojianet_ms import context
from luojianet_ms.train.serialization import load_checkpoint, load_param_into_net
from luojianet_ms.common import set_seed, Parameter
from src.luojia_detection.configuration.config import config
from src.luojia_detection.env.moxing_adapter import moxing_wrapper
from src.luojia_detection.datasets import create_dataset_inference
from src.luojia_detection.utils import bbox2result_1image
from src.luojia_detection.detectors import FasterRcnn_Infer

import cv2


def compute_IOU(rec1, rec2):  # cur_box, abox
    """
    Compute IoU of two boxes
    :param rec1: (x0,y0,x1,y1)   - xmin, ymin, xmax, ymax
    :param rec2: (x0,y0,x1,y1)
    :return: IOU number.
    """
    left_column_max  = max(rec1[0], rec2[0])  # 1213
    right_column_min = min(rec1[2], rec2[2])  # 1227
    up_row_max       = max(rec1[1], rec2[1])  # 422
    down_row_min     = min(rec1[3], rec2[3])  # 437
    # no overlap
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0, 0,  -1
    # overlap
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)

        return S_cross / S1, S_cross / S2, (S1 >= S2)


def fasterrcnn_inference(ckpt_path):
    """FasterRcnn evaluation."""
    if not os.path.isfile(ckpt_path):
        raise RuntimeError("CheckPoint file {} is not valid.".format(ckpt_path))
    ds, img_paths = create_dataset_inference()
    net = FasterRcnn_Infer(config)

    param_dict = load_checkpoint(ckpt_path)
    if config.device_target == "GPU":
        for key, value in param_dict.items():
            tensor = value.asnumpy().astype(np.float32)
            param_dict[key] = Parameter(tensor, key)
    for oldkey in list(param_dict.keys()):
        if oldkey.startswith(("backbone", "rcnn", "fpn_neck", "rpn_with_loss")):
            data = param_dict.pop(oldkey)
            newkey = 'network.' + oldkey
            param_dict[newkey] = data
    load_param_into_net(net, param_dict)

    net.set_train(False)
    device_type = "Ascend" if context.get_context("device_target") == "Ascend" else "Others"
    if device_type == "Ascend":
        net.to_float(mstype.float16)

    eval_iter = 0
    total = ds.get_dataset_size()
    total_time = 0
    total_num = 0
    cls_names = config.coco_classes

    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    max_num = 128

    if not os.path.exists(config.inference_save_dir):
        os.mkdir(config.inference_save_dir)
    txt_file = os.path.join(config.inference_save_dir, "result.txt")
    if os.path.exists(txt_file):
        os.remove(txt_file)
    for idx, data in enumerate(ds.create_dict_iterator(num_epochs=1)):
        eval_iter = eval_iter + 1
        img_data = data['image']
        img_metas = data['image_shape']
        img_path = img_paths[idx]

        #slide window det
        chip_size_h = config.img_height
        chip_size_w = config.img_width
        slide_size_h = chip_size_h // 2
        slide_size_w = chip_size_w // 2
        height, width = img_data.shape[2], img_data.shape[3]
        if height <= chip_size_h and width <= chip_size_w:
            h_stepnum, w_stepnum = 1, 1
        else:
            h_stepnum = int((height - chip_size_h) / slide_size_h) + 1
            w_stepnum = int((width - chip_size_w) / slide_size_w) + 1
            img_metas[0][0], img_metas[0][1] = chip_size_h, chip_size_w

        flt_all_bbox = []
        flt_all_label = []
        for i in range(w_stepnum):
            for j in range(h_stepnum):
                sub_img_data = img_data[:,:,j * slide_size_h:j * slide_size_h + chip_size_h, i * slide_size_w:i * slide_size_w + chip_size_w]
                start = time.time()
                output = net(sub_img_data, img_metas)
                end = time.time()
                # print("Iter {} cost time {}".format(eval_iter, end - start))
                total_time += (end - start)
                # output
                all_bbox = output[0]
                all_label = output[1]
                all_mask = output[2]

                all_bbox_squee = np.squeeze(all_bbox.asnumpy()[0, :, :])
                all_label_squee = np.squeeze(all_label.asnumpy()[0, :, :])
                all_mask_squee = np.squeeze(all_mask.asnumpy()[0, :, :])

                all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
                all_labels_tmp_mask = all_label_squee[all_mask_squee]

                if all_bboxes_tmp_mask.shape[0] > max_num:
                    inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
                    inds = inds[:max_num]
                    all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
                    all_labels_tmp_mask = all_labels_tmp_mask[inds]

                flt_bbox = bbox2result_1image(all_bboxes_tmp_mask, all_labels_tmp_mask, config.num_classes)
                for id in range(len(cls_names) - 1):
                    if all_labels_tmp_mask.shape[id] != 0:
                        for bbox in flt_bbox[id]:
                            if bbox[4] < 0.7:
                                continue
                            bbox[0] += i * slide_size_w
                            bbox[2] += i * slide_size_w
                            bbox[1] += j * slide_size_h
                            bbox[3] += j * slide_size_h
                            cur_bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), round(bbox[4], 3)]

                            if len(flt_all_bbox) != 0:
                                box_id = []
                                islarger_id = []
                                for id_num, abbox in enumerate(flt_all_bbox):
                                    s1_iou, s2_iou, s1lrs2 = compute_IOU(cur_bbox, abbox)
                                    if s2_iou >= 0.9:
                                        box_id.append(id_num)
                                        islarger_id.append(1)
                                    elif s1_iou > 0.6:  # 0.6
                                        box_id.append(id_num)
                                        islarger_id.append(s1lrs2)

                                if len(box_id) != 0:
                                    for k in range(len(box_id)):
                                        if islarger_id[k]:
                                            flt_all_bbox[box_id[k]] = cur_bbox

                                        elif not islarger_id[k]:
                                            pass
                                else:
                                    flt_all_bbox.append(cur_bbox)
                                    flt_all_label.append(id)

                            else:
                                flt_all_bbox.append(cur_bbox)
                                flt_all_label.append(id)

        ### save inference results
        img = cv2.imread(img_path)
        # img_shape = img.shape[:2]
        img_name = img_path.split("/")[-1]
        save_path = os.path.join(config.inference_save_dir, img_name)
        cnt = 0

        if len(flt_all_label) != 0:
            for bbox, label in zip(flt_all_bbox, flt_all_label):
                cnt += 1
                cls_name = cls_names[label + 1]
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), thickness=3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, '{:.2}, {}'.format(bbox[4], cls_name), (int(bbox[0]), int(bbox[1])), font, 0.4, (0, 0, 255), thickness=2)

                bbox_result = ' '
                for i in bbox[:4]:
                    bbox_result += str(int(i))
                    bbox_result += ' '
                with open(txt_file, 'a') as f:
                    f.write(img_name + ' ' + str(round(bbox[4], 3)) + bbox_result + '\n')
                f.close()
            cv2.imwrite(save_path, img)
        else:
            cv2.imwrite(save_path, img)
            pass
        total_num += cnt
        print(img_name, ", det num: ", cnt)
    print('det total num: ', total_num, ' time per image: ', total_time / total)


def modelarts_pre_process():
    pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def inference_fasterrcnn():
    """ eval_fasterrcnn """
    print("Start Inference!")
    fasterrcnn_inference(config.inference_checkpoint_path)  # change config-param names


if __name__ == '__main__':
    """
    python inference_fasterrcnn.py  --enable_infer \
                                    --infer_img_dir=./examples/inference_images/ \
                                    --infer_save_dir=./output_dir/inference_results/ \
                                    --infer_checkpoint_path=./pretrained_models/faster_rcnn-30_2253.ckpt
    """
    # set random seed
    set_seed(1)
    # set environment parameters
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=int(config.device_id))
    # run inference
    inference_fasterrcnn()
