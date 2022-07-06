# -*- coding:utf-8 -*-
import glob
import os

import cv2
import luojianet_ms
import luojianet_ms.dataset as ds

import luojianet_ms.dataset.vision.c_transforms as c_vision
from luojianet_ms.dataset.vision import Inter
import luojianet_ms.dataset.transforms.c_transforms as C


# import torch
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as F
# from PIL import Image
# import matplotlib.pyplot as plt
# import random
# import numpy as np


class DatasetGenerator(object):
    def __init__(self, cfg=None, test_mode=False):
        self.cfg = cfg
        self.rootdir = cfg.DATA.train_root if not test_mode else cfg.DATA.test_root
        self.targets = []
        self.names = []
        self.test_mode = test_mode
        self.all_path_list = []
        self.anno_list = cfg.DATA.train_list if not test_mode else cfg.DATA.test_list
        print(self.rootdir)
        with open(os.path.join(self.rootdir, '../', self.anno_list), 'r') as f:
            lines = f.readlines()
            for line in lines:
                if '\n' in line:
                    line = line[:-1]  # the txt should have a blank line with \n
                else:
                    line = line
                cur_pair = line.split(' ')
                img_name = str(cur_pair[0])
                cls = int(cur_pair[1])
                path_tmp = [img_name, str(cls)]
                self.all_path_list.append(path_tmp)
                self.names.append(img_name)
                self.targets.append(cls)
        print('in data_loader: Train data preparation done')

    def __getitem__(self, idx):
        # image
        img_path = os.path.join(self.rootdir, self.all_path_list[idx][0])
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.cfg.DATA.re_size, cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img/255.0  # if get image directly from DatasetGenerator object
        # label
        label = int(self.all_path_list[idx][1])

        return img, label

    def __len__(self):
        return len(self.all_path_list)


def data_aug(data_set, cfg, batch_size=128, num_parallel_workers=8, is_training=True):
    # resize settings
    resize_h = cfg.DATA.AUG.resize_h
    resize_w = cfg.DATA.AUG.resize_w

    # normalize settings
    mean = tuple(x/255.0 for x in cfg.DATA.AUG.mean)  # (122.67892/255, 116.66877/255, 104.00699/255)
    std = tuple(x for x in cfg.DATA.AUG.std)  # (1.0, 1.0, 1.0)

    # define map operations
    if is_training:
        transformations = [
            c_vision.Resize((resize_h, resize_w)),
            c_vision.RandomHorizontalFlip(prob=0.5),
            c_vision.RandomVerticalFlip(prob=0.5),
            c_vision.Normalize(mean=mean, std=std),
            c_vision.HWC2CHW()
        ]
    else:
        transformations = [
            c_vision.Resize((resize_h, resize_w)),
            c_vision.Normalize(mean=mean, std=std),
            c_vision.HWC2CHW(),
        ]

    type_cast_op = luojianet_ms.dataset.transforms.c_transforms.TypeCast(luojianet_ms.int32)
    img_type_cast_op = luojianet_ms.dataset.transforms.c_transforms.TypeCast(luojianet_ms.float32)

    # map operation
    data_set = data_set.map(operations=transformations, input_columns="image", num_parallel_workers=num_parallel_workers)
    data_set = data_set.map(operations=img_type_cast_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    data_set = data_set.batch(batch_size, drop_remainder=False)

    return data_set


def create_ml_dataset(dataset_generator, batch_size, is_training=True, shuffle=False, is_aug=True, num_parallel_workers=2):
    # convert dataset_generator to Generator_dataset object
    dataset = ds.GeneratorDataset(dataset_generator, ["image", "label"], shuffle=shuffle, num_parallel_workers=num_parallel_workers)
    # create aug dataset
    if is_aug:
        return data_aug(dataset, cfg=dataset_generator.cfg, batch_size=batch_size, num_parallel_workers=num_parallel_workers, is_training=is_training)
    else:
        return dataset


def create_mnist_dataset(data_path, batch_size=32, repeat_size=1, num_parallel_workers=1):
    mnist_ds = ds.MnistDataset(data_path)
    resize_height, resize_width = 28, 28
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    resize_op = c_vision.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = c_vision.Rescale(rescale_nml, shift_nml)
    rescale_op = c_vision.Rescale(rescale, shift)
    hwc2chw_op = c_vision.HWC2CHW()
    type_cast_op = C.TypeCast(luojianet_ms.int32)

    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(count=repeat_size)

    return mnist_ds


if __name__ == "__main__":
    from attrdict import AttrDict
    import yaml

    # set seed
    luojianet_ms.set_seed(8)

    # get config file
    cfg_path = '../../configs/ml_standard.yaml'
    with open(cfg_path, 'r') as f:
        cfg_ml = AttrDict(yaml.load(f, Loader=yaml.FullLoader))

    train_dataset = DatasetGenerator(cfg_ml)

    data_loader = create_ml_dataset(train_dataset, is_training=True, batch_size=cfg_ml.TRAIN.batch_size, is_aug=True, shuffle=False)

    print(data_loader.get_dataset_size())

    for idx, data in enumerate(data_loader.create_dict_iterator(num_epochs=1)):
        img = data["image"]
        label = data["label"]

        print('-----', img.shape)
        break
        # print('label:{}'.format(label))
    #
    #     plt.figure()
    #     plt.imshow(img.asnumpy()[0, :, :, :].squeeze().transpose(1, 2, 0))
    #     plt.show()
    #
    #     break



