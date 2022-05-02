import os
import numpy as np
import random
from PIL import Image
from mainnet import *
import luojianet_ms
from luojianet_ms.dataset.vision import Inter
import luojianet_ms.dataset.vision.c_transforms as c_vision
import luojianet_ms.dataset.vision.py_transforms as py_vision
import luojianet_ms.dataset as ds
import cv2
import glob

class changeDatasets:
    def __init__(self, data_path, is_Transforms: True):
        self.image1_dir = os.path.join(data_path, "A")
        self.image2_dir = os.path.join(data_path, "B")

        self.label1_dir = os.path.join(data_path, "building_A")
        self.label2_dir = os.path.join(data_path, "building_B")
        self.label3_dir = os.path.join(data_path, "label")

        if is_Transforms:
            self.toTensor = py_vision.ToTensor()

        self.files = os.listdir(self.label1_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image1_name = os.path.join(self.image1_dir, self.files[idx])
        image1 = Image.open(image1_name)

        image2_name = os.path.join(self.image2_dir, self.files[idx])
        image2 = Image.open(image2_name)

        label1_name = os.path.join(self.label1_dir, self.files[idx])
        label1 = Image.open(label1_name)

        label2_name = os.path.join(self.label2_dir, self.files[idx])
        label2 = Image.open(label2_name)

        label3_name = os.path.join(self.label3_dir, self.files[idx])
        label3 = Image.open(label3_name)


        image1 = self.toTensor(image1)
        image2 = self.toTensor(image2)

        label1 = self.toTensor(label1)
        label2 = self.toTensor(label2)
        label3 = self.toTensor(label3)
        
        image = np.concatenate([image1, image2], 0)
        label=np.concatenate([label1,label2,label3],0)
        # label = np.concatenate([np.expand_dims(label1[0, :, :], axis=0), np.expand_dims(label2[0, :, :], axis=0), np.expand_dims(label3[0, :, :], axis=0)], 0)
        
        return image, label

def create_Dataset(data_path, is_Transforms, batch_size, shuffle):

    CDatasets = changeDatasets(data_path, is_Transforms=True)
    Datasets = ds.GeneratorDataset(CDatasets, ["image", "label"], shuffle=shuffle)
    Datasets = Datasets.batch(batch_size=batch_size, drop_remainder=True)
    return Datasets,Datasets.get_dataset_size()


