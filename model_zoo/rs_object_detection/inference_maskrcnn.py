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

"""Evaluation for MaskRcnn"""
import os
import time
import numpy as np

from src.luojia_detection.configuration.config import config
from src.luojia_detection.env.moxing_adapter import moxing_wrapper
from src.luojia_detection.detectors import MaskRcnn_Infer
from src.luojia_detection.datasets import create_dataset_inference
from src.luojia_detection.utils import bbox2result_1image, get_seg_masks_inference

from luojianet_ms import context
from luojianet_ms.train.serialization import load_checkpoint, load_param_into_net
from luojianet_ms.common import set_seed, Parameter
import cv2
from shapely.geometry import Polygon


def compute_IOU_polygon(box1, box2):  # cur_box, abox
    box1 = np.array(box1).reshape(4, 2)
    poly1 = Polygon(box1).convex_hull

    box2 = np.array(box2).reshape(4, 2)
    poly2 = Polygon(box2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0.0
    else:
        inter_area = poly1.intersection(poly2).area

    w1 = max(box1[i][0] for i in range(4)) - min(box1[i][0] for i in range(4))
    h1 = max(box1[i][1] for i in range(4)) - min(box1[i][1] for i in range(4))
    S1 = w1 * h1

    w2 = max(box2[i][0] for i in range(4)) - min(box2[i][0] for i in range(4))
    h2 = max(box2[i][1] for i in range(4)) - min(box2[i][1] for i in range(4))
    S2 = w2 * h2

    return float(inter_area) / poly1.area, float(inter_area) / poly2.area, (S1 >= S2)


def maskrcnn_inference(ckpt_path):
    """MaskRcnn evaluation."""
    ds, img_paths = create_dataset_inference()

    net = MaskRcnn_Infer(config)
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

        # slide window det
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
        flt_all_rbbox = []
        for i in range(w_stepnum):
            for j in range(h_stepnum):
                sub_img_data = img_data[:, :, j * slide_size_h:j * slide_size_h + chip_size_h,
                               i * slide_size_w:i * slide_size_w + chip_size_w]
                start = time.time()
                # run net
                output = net(sub_img_data, img_metas)
                end = time.time()
                total_time += (end - start)

                # output
                all_bbox = output[0]
                all_label = output[1]
                all_mask = output[2]
                all_mask_fb = output[3]

                all_bbox_squee = np.squeeze(all_bbox.asnumpy()[0, :, :])
                all_label_squee = np.squeeze(all_label.asnumpy()[0, :, :])
                all_mask_squee = np.squeeze(all_mask.asnumpy()[0, :, :])
                all_mask_fb_squee = np.squeeze(all_mask_fb.asnumpy()[0, :, :, :])

                all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
                all_labels_tmp_mask = all_label_squee[all_mask_squee]
                all_mask_fb_tmp_mask = all_mask_fb_squee[all_mask_squee, :, :]

                if all_bboxes_tmp_mask.shape[0] > max_num:
                    inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
                    inds = inds[:max_num]
                    all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
                    all_labels_tmp_mask = all_labels_tmp_mask[inds]
                    all_mask_fb_tmp_mask = all_mask_fb_tmp_mask[inds]

                flt_bbox = bbox2result_1image(all_bboxes_tmp_mask, all_labels_tmp_mask, config.num_classes)
                flt_mask = get_seg_masks_inference(all_mask_fb_tmp_mask, all_bboxes_tmp_mask, all_labels_tmp_mask, img_metas[0],
                                             True, config.num_classes)
                for id in range(len(cls_names) - 1):
                    if len(flt_bbox[id]) != 0:
                        for bbox, amask in zip(flt_bbox[id], flt_mask[id]):
                            if bbox[4] < 0.7:
                                continue
                            bbox[0] += i * slide_size_w
                            bbox[2] += i * slide_size_w
                            bbox[1] += j * slide_size_h
                            bbox[3] += j * slide_size_h
                            cur_bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), round(bbox[4], 3)]
                            cur_rbbox = get_mask_box(amask, height, width, i * slide_size_w, j * slide_size_h).reshape(4,2)

                            if len(flt_all_rbbox) != 0:
                                box_id = []
                                islarger_id = []
                                for id_num, amaskbox in enumerate(flt_all_rbbox):
                                    s1_iou, s2_iou, s1lrs2 = compute_IOU_polygon(cur_rbbox, amaskbox)
                                    if s2_iou >= 0.9:
                                        box_id.append(id_num)
                                        islarger_id.append(1)
                                    elif s1_iou > 0.6:  # 0.6
                                        box_id.append(id_num)
                                        islarger_id.append(s1lrs2)

                                if len(box_id) != 0:
                                    for k in range(len(box_id)):
                                        if islarger_id[k]:
                                            flt_all_rbbox[box_id[k]] = cur_rbbox
                                            flt_all_bbox[box_id[k]] = cur_bbox

                                        elif not islarger_id[k]:
                                            pass
                                else:
                                    flt_all_bbox.append(cur_bbox)
                                    flt_all_label.append(id)
                                    flt_all_rbbox.append(cur_rbbox)

                            else:
                                flt_all_bbox.append(cur_bbox)
                                flt_all_label.append(id)
                                flt_all_rbbox.append(cur_rbbox)

        ### save inference results
        img = cv2.imread(img_path)
        img_name = img_path.split("/")[-1]
        save_path = os.path.join(config.inference_save_dir, img_name)
        cnt = 0

        # visualize
        if len(flt_all_label) != 0:
            for bbox, label in zip(flt_all_bbox, flt_all_label):
                cnt += 1
                cls_name = cls_names[label + 1]
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255),
                              thickness=3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, '{:.2}, {}'.format(bbox[4], cls_name), (int(bbox[0]), int(bbox[1])), font, 0.4,
                            (0, 0, 255), thickness=1)
                bbox_result = ' '
                for i in bbox[:4]:
                    bbox_result += str(int(i))
                    bbox_result += ' '
                with open(txt_file, 'a') as f:
                    f.write(img_name + ' ' + str(round(bbox[4], 3)) + bbox_result + '\n')
                f.close()
            for rbbox in flt_all_rbbox:
                cv2.polylines(img, [rbbox], 1, (0, 255, 0), thickness=3)
            cv2.imwrite(save_path, img)
        else:
            cv2.imwrite(save_path, img)
            pass
        total_num += cnt
        print(img_name, ", det num: ", cnt)
    print('det total num: ', total_num, ' time per img: ', total_time / total)


def modelarts_process():
    pass


def get_mask_box(mask, img_h, img_w, x_append=0, y_append=0):
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours[0]))
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


@moxing_wrapper(pre_process=modelarts_process)
def inference_():
    print("Start Inference!")
    maskrcnn_inference(config.inference_checkpoint_path)


if __name__ == '__main__':
    """
    python inference_maskrcnn.py  --enable_infer \
                                --infer_img_dir=./examples/inference_images/ \
                                --infer_save_dir=./output_dir/inference_results/ \
                                --infer_checkpoint_path=./pretrained_models/mask_rcnn-30_2253.ckpt
    """
    # set random seed, if necessary
    set_seed(1)

    # set environment
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=int(config.device_id))

    # run inference
    inference_()
