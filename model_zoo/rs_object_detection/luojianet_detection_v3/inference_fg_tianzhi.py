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

import os
import time
import numpy as np

from src.luojianet_ml_tool.inference import ml_inference, get_model
from src.luojia_detection.configuration.config import config
from src.luojia_detection.env.moxing_adapter import moxing_wrapper
from src.luojia_detection.detectors import MaskRcnn_Infer
from src.luojia_detection.utils import bbox2result_1image, get_seg_masks_inference
from luojianet_ms import context, Tensor
import luojianet_ms.common.dtype as mstype
from luojianet_ms.train.serialization import load_checkpoint, load_param_into_net
from luojianet_ms.common import set_seed, Parameter
import cv2
import glob
from shapely.geometry import Polygon

def imnormalize_img(img, img_shape):
    mean = np.asarray([123.675, 116.28, 103.53])
    std = np.asarray([58.395, 57.12, 57.375])
    img_data = img.copy().astype(np.float32)
    cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB, img_data)  # inplace
    cv2.subtract(img_data, np.float64(mean.reshape(1, -1)), img_data)  # inplace
    cv2.multiply(img_data, 1 / np.float64(std.reshape(1, -1)), img_data)  # inplace
    img_data = img_data.astype(np.float32)
    return img_data, img_shape

def transpose_img(img, img_shape):
    """transpose operation for image"""
    img_data = img.transpose(2, 0, 1).copy()
    img_data = img_data.astype(np.float32)
    img_shape = img_shape.astype(np.float32)
    return img_data, img_shape

def preprocess_img(image):
    """Preprocess function for dataset."""
    def _infer_data(image_bgr, image_shape):
        image_shape = image_shape[:2]
        image_shape = np.append(image_shape, (1.0, 1.0))

        image_data, image_shape = imnormalize_img(image_bgr, image_shape)

        image_data, image_shape = transpose_img(image_data, image_shape)
        return image_data, image_shape

    def _data_aug(image):
        """Data augmentation function."""
        image_bgr = image.copy()
        image_bgr[:, :, 0] = image[:, :, 2]
        image_bgr[:, :, 1] = image[:, :, 1]
        image_bgr[:, :, 2] = image[:, :, 0]
        image_shape = image_bgr.shape[:2]

        return _infer_data(image_bgr, image_shape)

    return _data_aug(image)

def cur_result_str_all(cls_list, scores_list, boxes_list, rboxes_list, img_name):
    classes = cls_list
    bbox = boxes_list
    scores = scores_list
    rbbox = rboxes_list

    cur_out = []
    cur_out_rbox = []

    for i in range(len(classes)):
        str_out = []
        str_out_rbox = []
        cls_id = int(classes[i])
        box = bbox[i]
        rbox = rbbox[i].reshape(-1)
        confidence = scores[i]

        str_out.append(img_name)
        str_out.append('{:.4}'.format(confidence))

        str_out.append(str(int(box[0])))
        str_out.append(str(int(box[1])))
        str_out.append(str(int(box[2])))
        str_out.append(str(int(box[3])))

        str_out.append(str(cls_id))
        cur_out.append(str_out)

        str_out_rbox.append(img_name)
        str_out_rbox.append('{:.4}'.format(confidence))

        str_out_rbox.append(str(int(rbox[0])))
        str_out_rbox.append(str(int(rbox[1])))
        str_out_rbox.append(str(int(rbox[2])))
        str_out_rbox.append(str(int(rbox[3])))
        str_out_rbox.append(str(int(rbox[4])))
        str_out_rbox.append(str(int(rbox[5])))
        str_out_rbox.append(str(int(rbox[6])))
        str_out_rbox.append(str(int(rbox[7])))

        str_out_rbox.append(str(cls_id))
        cur_out_rbox.append(str_out_rbox)

    return cur_out, cur_out_rbox


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


def compute_IOU(rec1, rec2):  # cur_box, abox
    """
    Compute IoU of two boxes
    :param rec1: (x0,y0,x1,y1)   - xmin, ymin, xmax, ymax
    :param rec2: (x0,y0,x1,y1)
    :return: IOU number.
    """
    left_column_max  = max(rec1[0], rec2[0])
    right_column_min = min(rec1[2], rec2[2])
    up_row_max       = max(rec1[1], rec2[1])
    down_row_min     = min(rec1[3], rec2[3])
    # no overlap
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0, 0,  -1
    # overlap
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)

        return S_cross / S1, S_cross / S2, (S1 >= S2)


def maskrcnn_inference(ckpt_path):
    img_paths = glob.glob(config.inference_img_dir + '*.png')

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

    total = len(img_paths)
    total_time = 0
    total_num = 0
    cls_names = config.coco_classes
    res_dict = {}
    res_dict_rbox = {}
    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    max_num = 128

    if not os.path.exists(config.inference_save_dir):
        os.mkdir(config.inference_save_dir)

    hbb_dir = os.path.join(config.inference_save_dir, 'hbb_result/')
    if not os.path.exists(hbb_dir):
        os.mkdir(hbb_dir)

    obb_dir = os.path.join(config.inference_save_dir, 'obb_result/')
    if not os.path.exists(obb_dir):
        os.mkdir(obb_dir)

    cfg, inference_model, train_vectors, name_list, label_list, labels_to_indices = get_model(
        cfg_path=config.ml_cfg_path,
        checkpoint_path=config.ml_checkpoint_path)

    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)  # h, w, c
        img_name = img_path.split("/")[-1]
        print("{} / {}, img_name: {}".format(idx + 1, total, img_name))
        # slide window det
        chip_size_h = config.img_height
        chip_size_w = config.img_width
        slide_size_h = chip_size_h // 2
        slide_size_w = chip_size_w // 2
        height, width, channel = img.shape
        if height < chip_size_h or width < chip_size_w:
            right_padding = max(chip_size_w - width, 0)
            down_padding = max(chip_size_h - height, 0)
            img = cv2.copyMakeBorder(img, 0, down_padding, 0, right_padding, cv2.BORDER_CONSTANT,
                                     value=[0, 0, 0])
            height, width, channel = img.shape

        h_stepnum = int((height - chip_size_h) / slide_size_h) + 1
        w_stepnum = int((width - chip_size_w) / slide_size_w) + 1
        if (height - (h_stepnum - 1) * slide_size_h - chip_size_h > 0):
            down_padding = slide_size_h - (height - (h_stepnum - 1) * slide_size_h - chip_size_h)
            img = cv2.copyMakeBorder(img, 0, down_padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            h_stepnum += 1
        if (width - (w_stepnum - 1) * slide_size_w - chip_size_w > 0):
            right_padding = slide_size_w - (width - (w_stepnum - 1) * slide_size_w - chip_size_w)
            img = cv2.copyMakeBorder(img, 0, 0, 0, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            w_stepnum += 1

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # c, h, w
        img, img_metas = preprocess_img(img)
        height, width = img.shape[1], img.shape[2]
        img_metas[0], img_metas[1] = chip_size_h, chip_size_w

        img = np.expand_dims(img, axis=0)
        img_metas = np.expand_dims(img_metas, axis=0)
        img_metas = Tensor(img_metas, mstype.float32)

        flt_all_bbox = []
        flt_all_label = []
        flt_all_rbbox = []
        cls_list = []
        for i in range(w_stepnum):
            for j in range(h_stepnum):
                sub_img_data = img[:, :, j * slide_size_h:j * slide_size_h + chip_size_h,
                               i * slide_size_w:i * slide_size_w + chip_size_w]
                sub_img_data = Tensor(sub_img_data, mstype.float32)
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
                                for id_num, abox in enumerate(flt_all_bbox):
                                    s1_iou, s2_iou, s1lrs2 = compute_IOU(cur_bbox, abox)
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
        # ml_inference
        img = cv2.imread(img_path)
        (h, w, _) = img.shape
        score_list = []
        box_list = []
        for box, rbox in zip(flt_all_bbox, flt_all_rbbox):
            score_list.append(box[4])
            box_list.append([box[0], box[1], box[2], box[3]])
            rbox_xmin, rbox_xmax = box[0], box[2]
            rbox_ymin, rbox_ymax = box[1], box[3]

            leng_w = int(rbox_xmax - rbox_xmin)
            leng_h = int(rbox_ymax - rbox_ymin)
            length = (max(leng_w, leng_h) // 2) + 1
            mid_x = int(rbox_xmax + rbox_xmin) // 2
            mid_y = int(rbox_ymax + rbox_ymin) // 2
            top = max(mid_y - length, 0)
            bottom = min(mid_y + length, h)
            left = max(mid_x - length, 0)
            right = min(mid_x + length, w)
            ap_img = img.copy()[top:bottom, left:right, :]
            ml_pred = ml_inference(cfg, ap_img.copy(), inference_model, train_vectors, name_list, label_list, rank_k=cfg.TEST.rank_k)
            cls_list.append(ml_pred)
        ### save inference results
        total_num += 1
        print(cls_list)
        # draw img
        img_draw = img.copy()
        for aconf, abox, cls in zip(score_list, box_list, cls_list):
            cv2.rectangle(img_draw, (abox[0], abox[1]), (abox[2], abox[3]), (0, 0, 255), thickness=3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_draw, '{:.2}, {}'.format(aconf, cls), (abox[0], abox[1]), font, 0.5,
                        (255, 255, 255), thickness=2)

        for amaskbox in flt_all_rbbox:
            amaskbox = np.array(amaskbox).reshape(4,2)
            cv2.polylines(img_draw, [amaskbox], 1, (0, 255, 0), thickness=3)
        cv2.imwrite(config.inference_save_dir + img_name.split('.')[0] + '.png', img_draw)

        cur_lines, cur_lines_rbbox = cur_result_str_all(cls_list, score_list, box_list, flt_all_rbbox, img_name)
        for line, line_rbox in zip(cur_lines, cur_lines_rbbox):
            if line[-1] not in res_dict.keys():
                res_dict[line[-1]] = [line]
                res_dict_rbox[line[-1]] = [line_rbox]
            else:
                res_dict[line[-1]].append(line)
                res_dict_rbox[line[-1]].append(line_rbox)

    print('-' * 10, 'res_dict', '-' * 10)
    for k, v in res_dict.items():
        txt_save_path = hbb_dir + str(k) + '.txt'
        if os.path.exists(txt_save_path):
            file = open(txt_save_path, 'w')
        else:
            file = open(txt_save_path, 'a')

        for line in v:
            line_str = [x + ' ' for x in line[:-1]]
            file.writelines(line_str)
            file.writelines('\r\n')

        file.close()

    for k, v in res_dict_rbox.items():
        txt_save_path_rbox = obb_dir + str(int(k)) + '.txt'
        if os.path.exists(txt_save_path_rbox):
            file_rbox = open(txt_save_path_rbox, 'w')
        else:
            file_rbox = open(txt_save_path_rbox, 'a')

        for line in v:
            line_str = [x + ' ' for x in line[:-1]]
            file_rbox.writelines(line_str)
            file_rbox.writelines('\r\n')

        file_rbox.close()

    print("Average Time: {:.2f} s /img".format(total_time / total_num))


def modelarts_process():
    pass


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

@moxing_wrapper(pre_process=modelarts_process)
def inference_():

    print("Start Inference!")
    maskrcnn_inference(config.inference_checkpoint_path)


if __name__ == '__main__':
    """
        python inference_fg_tianzhi.py  --enable_infer \
                                    --config_path=./configs/mask_rcnn_r152_fpn_ml_inference.yaml \
                                    --infer_img_dir=./examples/inference_images/ \
                                    --infer_save_dir=./output_dir/inference_results/ \
                                    --infer_checkpoint_path=./pretrained_models/mask_rcnn-30_2253.ckpt \
                                    --ml_infer_cfg_path=./configs/ml_standard.yaml \
                                    --ml_infer_checkpoint_path=./output_dir/ml_mini_dataset/gt_det_12-10_5.ckpt
    """
    # set random seed, if necessary
    set_seed(1)

    # set environment
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=int(config.device_id))

    # run inference
    inference_()
