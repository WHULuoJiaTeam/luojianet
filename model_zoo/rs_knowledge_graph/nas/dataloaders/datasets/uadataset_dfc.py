"""Data Transformations and pre-processing."""

from __future__ import print_function, division

import os
import warnings
import random
import cv2
import torchvision.transforms.functional as TF
import numpy as np
import torch
import rasterio
from rasterio.enums import Resampling
from PIL import Image
from torch.utils.data import Dataset

class UADataset(Dataset):
    """Custom Pascal VOC"""

    def __init__(self, stage, data_file, data_dir, transform_trn=None, transform_val=None, transform_test=None):
        """
        Args:
            data_file (string): Path to the data file with annotations.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(data_file, "rb") as f:
            datalist = f.readlines()
        try:
            # self.datalist = [
            #     (k, k.replace('image/', 'label/').replace('img.tif', 'label.png'))
            #     for k in map(
            #         lambda x: x.decode("utf-8").strip("\n").strip("\r"), datalist
            #     )
            # ]
            self.datalist = [
                (k[0], k[1])
                for k in map(
                    lambda x: x.decode("utf-8").strip("\n").strip("\r").split("\t"), datalist
                )
            ]
        except ValueError:  # Adhoc for test.
            self.datalist = [
                (k, k) for k in map(lambda x: x.decode("utf-8").strip("\n"), datalist)
            ]
        self.split = stage

        self.mean = (0.40781063, 0.44303973, 0.35496944)
        self.std = (0.3098623, 0.2442191, 0.22205387)
        self.all_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.valid_classes = [255, 0, 1, 2, 3, 4, 5, 6, 255, 255, 7, 8, 9, 10, 11, 255]
        self.class_map = dict(zip(self.all_classes, self.valid_classes))
        self.root_dir = data_dir
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.transform_test = transform_test
        self.stage = stage



    def set_config(self, crop_size, resize_side):
        self.transform_trn.transforms[0].resize_side = resize_side
        self.transform_trn.transforms[2].crop_size = crop_size

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.datalist[idx][0])
        msk_name = os.path.join(self.root_dir, self.datalist[idx][1])

        def read_image(x):
            img_arr = np.array(Image.open(x), dtype=np.uint8)
            if len(img_arr.shape) == 2:  # grayscale
                img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
            return img_arr

        slp_path = img_name.replace('320img512', '320slopecrop').replace('E080', 'E080slope')
        arf_path = img_name.replace('320img512', '320arfcrop').replace('E080', 'E080arf')

        with rasterio.open(img_name) as image:
            _img = image.read().astype(np.float32).transpose(1, 2, 0)
        with rasterio.open(msk_name) as label:
            _tmp = label.read()
            _tmp = _tmp.squeeze()
        with rasterio.open(slp_path) as slope:
            # _slp = slope.read(out_shape=(1, _img.shape[0], _img.shape[1]), resampling=Resampling.bilinear)
            _slp = slope.read()
        with rasterio.open(arf_path) as arf:
            # _arf = arf.read(out_shape=(1, _img.shape[0], _img.shape[1]), resampling=Resampling.bilinear)
            _arf = arf.read()

        _img /= 255.0
        _img -= self.mean
        _img /= self.std
        _img = _img.transpose(2, 0, 1)
        _img = torch.from_numpy(_img).float()
        _tmp = torch.from_numpy(_tmp).float()
        _slp /= 90.0
        _slp = torch.from_numpy(_slp).float()
        _arf /= 270.0
        _arf = torch.from_numpy(_arf).float()

        _img = torch.cat((_img, _slp, _arf), 0)

        # mask = self.encode_segmap(mask).astype(np.float32)
        # if img_name != msk_name:
        #     assert len(mask.shape) == 2, "Masks must be encoded without colourmap"
        if 'train' in self.split:
            a = random.random()
            if a < 0.5:
                _img = TF.hflip(_img)
                _tmp = TF.hflip(_tmp)
            b = random.random()
            if b < 0.5:
                _img = TF.vflip(_img)
                _tmp = TF.vflip(_tmp)
            c = random.random()
            if c < 0.5:
                _img = TF.rotate(_img, 90)
                _tmp = TF.rotate(_tmp.unsqueeze(0), 90).squeeze()

        sample = {'image': _img, 'mask': _tmp}
        return sample

    def encode_segmap(self, mask):
        for _validc in self.all_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
