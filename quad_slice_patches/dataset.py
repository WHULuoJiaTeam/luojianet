##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: ZhangZhan
## Wuhan University
## zhangzhanstep@whu.edu.cn
## Copyright (c) 2022
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import cv2

import luojianet_ms.dataset as ds
from luojianet_ms.communication.management import get_rank

from settings import get_args
# from geobject import get_objects


def pad(ignore_label, image_list, label_list):
    max_searchsize = 1024
    for i in range(len(image_list)):
        h, w = image_list[i].shape[:2]
        pad_h = max(max_searchsize - h, 0)
        pad_w = max(max_searchsize - w, 0)
        if pad_h > 0 or pad_w > 0:
            image_list[i] = cv2.copyMakeBorder(image_list[i], 0, pad_h, 0, pad_w,
                                               cv2.BORDER_CONSTANT, value=0.)
            label_list[i] = cv2.copyMakeBorder(label_list[i], 0, pad_h, 0, pad_w,
                                               cv2.BORDER_CONSTANT, value=ignore_label)
    return image_list, label_list


def normalize(mean, std, image_list, label_list):
    for i in range(len(image_list)):
        image_list[i] = (image_list[i] - mean) / std
    return image_list, label_list


# label should be encoded to int-datatype (e.g., 1, 2, 3...) first.
def get_file_list(split):
    args = get_args()

    id_list = os.path.join('datalist', split + '.txt')
    id_list = tuple(open(id_list, 'r'))
    if split == 'val':
        image_files = [os.path.join(args.val_image_dir, id_.rstrip()) for id_ in id_list]
        label_files = [os.path.join(args.val_label_dir, id_.rstrip()) for id_ in id_list]
    else:
        image_files = [os.path.join(args.train_image_dir, id_.rstrip()) for id_ in id_list]
        label_files = [os.path.join(args.train_label_dir, id_.rstrip()) for id_ in id_list]
    return image_files, label_files


class BaseDataset:
    def __init__(self, split):
        self.image_files, self.label_files = get_file_list(split)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        label_path = self.label_files[index]
        return self._get_item(image_path, label_path)

    def _get_item(self, image_path, label_path):
        raise NotImplementedError


class TrainDataset(BaseDataset):
    def __init__(self,
                 n_classes,
                 ignore_label,
                 image_mean,
                 image_std,
                 device_num,
                 split='train'):
        self.n_classes = n_classes
        self.ignore_label = ignore_label
        self.image_mean = image_mean
        self.image_std = image_std
        self.device_num = device_num
        super(TrainDataset, self).__init__(split)

    def preprocess_data(self, image_list, label_list):
        image_list, label_list = normalize(self.image_mean, self.image_std, image_list, label_list)
        image_list, label_list = pad(self.ignore_label, image_list, label_list)
        return image_list, label_list

    def _get_item(self, image_path, label_path):
        image_list = []
        label_list = []
        for i in range(3):
            image_list[i].append(cv2.imread('GDataset/train/image/GF2_PMS1__L1A0000564539-MSS1.tif'))
            label_list[i].append(cv2.imread('GDataset/train/label/GF2_PMS1__L1A0000564539-MSS1_label.tif'))
        # Get the minimum bounding rectangle data of one-specified class, max_size of data is 1024x1024.
        # image_objects, label_objects = get_objects(device_num=self.device_num, rank_id=get_rank(),
        #                                            image_path=image_path, label_path=label_path,
        #                                            n_classes=self.n_classes, ignore_label=self.ignore_label,
        #                                            block_size=4096, max_searchsize=1024, min_searchsize=16)
        # image_list, label_list = self.preprocess_data(image_list=image_objects, label_list=label_objects)
        return image_list, label_list


def get_dataset(split='train', repeat=1):
    args = get_args()

    if split == 'val':
        raise NotImplementedError
    train_dt = TrainDataset(args.n_classes, args.ignore_label, args.image_mean, args.image_std, args.device_num)
    dataset = ds.GeneratorDataset(source=train_dt, column_names=["image_list", "label_list"],
                                  shuffle=False, num_parallel_workers=args.num_parallel_workers)
    dataset = dataset.batch(args.batch_size, drop_remainder=True)
    dataset = dataset.repeat(repeat)
    return dataset


def test_dt():
    dataset = get_dataset(split='train', repeat=1)
    dataset_size = dataset.get_dataset_size()
    for data in dataset.create_dict_iterator():
        image_list, label_list = data["image_list"], data["label_list"]
        print(image_list.shape, label_list.shape)


if __name__ == '__main__':
    test_dt()