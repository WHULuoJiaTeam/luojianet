import os
import glob
import random
from re import A
import numpy as np
from luojianet_ms.common import dtype as mstype
import luojianet_ms.dataset.vision.c_transforms as c_vision
import luojianet_ms.dataset.vision.py_transforms as py_vision
from PIL import Image as pI
import luojianet_ms.dataset as ds
import luojianet_ms.dataset.transforms.c_transforms as C2
from config import config 

class Dataset:
    '''Dataset'''
    def __init__(self, data_path, aug=True):
        super(Dataset, self).__init__()
        self.img1_paths = glob.glob(os.path.join(data_path, "A", "*"))
        self.img2_paths = glob.glob(os.path.join(data_path, "B", "*"))
        self.img1_label_path = glob.glob(os.path.join(data_path, "building_A", "*"))
        self.img2_label_path = glob.glob(os.path.join(data_path, "building_B", "*"))
        self.mask_paths = glob.glob(os.path.join(data_path, "label", "*"))
        self.aug = aug
        self.toTensor = py_vision.ToTensor()

    def __getitem__(self, idx):
        img1_path = self.img1_paths[idx]
        img2_path = self.img2_paths[idx]
        img1_label_path = self.img1_label_path[idx]
        img2_label_path = self.img2_label_path[idx]
        mask_path = self.mask_paths[idx]
       
        image1=pI.open(img1_path)
        image2=pI.open(img2_path)
        image1_label=pI.open(img1_label_path)
        image2_label=pI.open(img2_label_path)
        mask=pI.open(mask_path)
        
        image1 = self.toTensor(image1)
        image2 = self.toTensor(image2)
        image1_label = self.toTensor(image1_label)
        image2_label = self.toTensor(image2_label)
        mask = self.toTensor(mask)
        return [image1, image2], [image1_label, image2_label, mask]

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

#测试data_loader
# if __name__ == '__main__':
#     # dataset=Dataset(config.dataset_path, config.aug)
#     # image1=dataset.__getitem__(2)[0][0]
#     # image2=dataset.__getitem__(2)[0][1]
#     # image3=dataset.__getitem__(2)[1][0]
#     # image4=dataset.__getitem__(2)[1][1]
#     # image5=dataset.__getitem__(2)[1][2]
#     CDataset,_=create_Dataset(config.dataset_path, config.aug,config.batch_size,True)
#     # Dataset1=changeDatasets(img1_paths,img2_paths,img1_label_path,img2_label_path,mask_paths,True)
#     # Dataset1.__getitem__(0)
#     iterator=CDataset.create_dict_iterator()
#     for i in iterator:
#         print(i["image"].shape)

