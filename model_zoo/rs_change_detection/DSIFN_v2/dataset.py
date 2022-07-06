import os
import glob
import random
from re import A
import numpy as np
from luojianet_ms.common import dtype as mstype
from PIL import Image as pI
import luojianet_ms.dataset as ds
import luojianet_ms.dataset.transforms.c_transforms as C2
import luojianet_ms.dataset.vision.c_transforms as c_vision
import luojianet_ms.dataset.vision.py_transforms as py_vision
from luojianet_ms.dataset.vision import Border

class Dataset:
    '''Dataset'''
    def __init__(self, data_path, aug=True):
        super(Dataset, self).__init__()
        self.img1_paths = glob.glob(os.path.join(data_path, "A", "*"))
        self.img2_paths = glob.glob(os.path.join(data_path, "B", "*"))
        self.mask_paths = glob.glob(os.path.join(data_path, "label", "*"))
        self.aug = aug
        self.toTensor = py_vision.ToTensor()
        self.patch_size = 512

    def random_position(self, crop_h, crop_w, h, w):
        if crop_w != w and crop_h != h:
            y = np.random.randint(0, h - crop_h - 1)
            x = np.random.randint(0, w - crop_w - 1)

            return x, y
        else:
            return 0, 0
    
    def __getitem__(self, idx):
        img1_path = self.img1_paths[idx]
        img2_path = self.img2_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image1 = pI.open(img1_path)
        image2 = pI.open(img2_path)
        mask   = pI.open(mask_path)
        
        # image1 = self.toTensor(image1)
        # image2 = self.toTensor(image2)
        # mask   = self.toTensor(mask)
        image1 = np.asarray(image1)
        image2 = np.asarray(image2)
        mask   = np.asarray(mask)

        x, y = self.random_position(self.patch_size, self.patch_size, image1.shape[0], image1.shape[1])
        
        image1 = image1[y:y + self.patch_size, x:x + self.patch_size, :]
        image2 = image2[y:y + self.patch_size, x:x + self.patch_size, :]
        mask = mask[y:y + self.patch_size, x:x + self.patch_size]
        mask = mask/255

        image1 = image1.transpose(2, 0, 1)
        image2 = image2.transpose(2, 0, 1)
        mask = np.expand_dims(mask, 0)

        images = [image1, image2]
        
        return images, mask

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
