import os
import glob
import random
from re import A
import numpy as np
from skimage.io import imread
from skimage import color
from luojianet_ms.common import dtype as mstype
from PIL import Image as pI
import luojianet_ms.dataset as ds
import luojianet_ms.dataset.transforms.c_transforms as C2
import luojianet_ms.dataset.vision.c_transforms as c_vision
import luojianet_ms.dataset.vision.py_transforms as py_vision

class Dataset:
    '''Dataset'''
    def __init__(self, data_path, aug=True):
        super(Dataset, self).__init__()
        self.img1_paths = glob.glob(os.path.join(data_path, "A", "*"))
        self.img2_paths = glob.glob(os.path.join(data_path, "B", "*"))
        self.mask_paths = glob.glob(os.path.join(data_path, "label", "*"))
        # self.aug = aug
        self.toTensor()=py_vision.toTensor()

    def __getitem__(self, idx):
        img1_path = self.img1_paths[idx]
        img2_path = self.img2_paths[idx]
        mask_path = self.mask_paths[idx]

        # image1 = imread(img1_path)
        # image2 = imread(img2_path)
        # mask = imread(mask_path)

        # image1 = image1.astype('float32') / 255
        # image2 = image2.astype('float32') / 255
        # mask = mask.astype('float32') / 255

        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         image1 = image1[:, ::-1, :].copy()
        #         image2 = image2[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         image1 = image1[::-1, :, :].copy()
        #         image2 = image2[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()

        # image1 = color.gray2rgb(image1)
        # image2 = color.gray2rgb(image2)
        # image1 = np.transpose(image1,(2,0,1))
        # image2 = np.transpose(image2,(2,0,1))
        # mask = mask[np.newaxis, :, :]

        image1=pI.open(img1_path)
        image2=pI.open(img2_path)
        mask=pI.open(mask_path)
        
        image1 = self.toTensor(image1)
        image2 = self.toTensor(image2)
        mask = self.toTensor(mask)

        return [image1, image2], mask

    def __len__(self):
        return len(self.img1_paths)


def create_Dataset(data_path, aug, batch_size, shuffle):

    dataset = Dataset(data_path, aug)
    type_ops = C2.TypeCast(mstype.float32)

    data_set = ds.GeneratorDataset(dataset, column_names=["image","mask"], shuffle=shuffle)
    data_set = data_set.map(input_columns=["image"], operations = type_ops)
    data_set = data_set.map(input_columns=["mask"], operations = type_ops)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set, data_set.get_dataset_size()
