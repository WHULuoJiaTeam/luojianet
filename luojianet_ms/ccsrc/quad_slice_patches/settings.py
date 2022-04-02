# Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
# Copyright 2021 Huawei Technologies Co., Ltd
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

import argparse


def get_args():
    parser = argparse.ArgumentParser(description='LuoJiaNET land use classification Example')

    # datalist settings
    parser.add_argument('--train_image_dir', type=str, default='GDataset/train/image/', help='path to train image')
    parser.add_argument('--train_label_dir', type=str, default='GDataset/train/label/', help='path to train label')
    parser.add_argument('--val_image_dir', type=str, default='GDataset/val/image/', help='path to val image')
    parser.add_argument('--val_label_dir', type=str, default='GDataset/val/label/', help='path to val label')

    # dataset settings
    parser.add_argument('--n_classes', type=int, default=6, help='number of classes')
    parser.add_argument('--ignore_label', type=int, default=255, help='ignore label value')
    parser.add_argument('--image_mean', type=list, default=[103.53, 116.28, 123.675], help='image mean')
    parser.add_argument('--image_std', type=list, default=[57.375, 57.120, 58.395], help='image std')
    parser.add_argument('--num_parallel_workers', type=int, default=1, help='number of subprocesses to process data')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')

    # network settings
    parser.add_argument('--model_name', type=str, default='DeepLabV3plus_s16', help='model name')
    parser.add_argument('--output_stride', type=int, default=16, help='output stride, choices=[8, 16]')
    parser.add_argument('--freeze_bn', action='store_true', help='freeze bn')

    # training settings
    parser.add_argument('--device_target', type=str, default='GPU', choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    parser.add_argument('--is_distributed', action='store_true', default=True, help='distributed training')
    parser.add_argument('--device_number', type=int, default=8, help='world size of distributed, acquired automatically')

    parser.add_argument('--ckpt_file', type=str, default='', help='path to ckpt model')
    parser.add_argument('--model_dir', type=str, default='models', help='where training log and model saved')
    parser.add_argument('--save_steps', type=int, default=400, help='steps interval for saving')
    parser.add_argument('--keep_checkpoint_max', type=int, default=200, help='max checkpoint for saving')

    parser.add_argument('--train_epochs', type=int, default=300, help='epoch')
    parser.add_argument('--lr_type', type=str, default='cos', help='type of learning rate')
    parser.add_argument('--base_lr', type=float, default=0.02, help='base learning rate')
    parser.add_argument('--lr_decay_step', type=int, default=40000, help='learning rate decay step')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--loss_scale', type=float, default=3072.0, help='loss scale')

    args = parser.parse_args()
    return args