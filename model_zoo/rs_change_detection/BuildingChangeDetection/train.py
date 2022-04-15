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
from luojianet_ms import context,nn
from luojianet_ms.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor

context.set_context(device_target='GPU')

class changeDatasets:
    def __init__(self, image1_dir, image2_dir, label1_dir, label2_dir, label3_dir, is_Transforms: True):
        self.image1_dir = image1_dir
        self.image2_dir = image2_dir

        self.label1_dir = label1_dir
        self.label2_dir = label2_dir
        self.label3_dir = label3_dir


        if is_Transforms:
            self.tx = py_vision.ToTensor()

            self.lx = py_vision.ToTensor()

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

        image1 = self.tx(image1)
        image2 = self.tx(image2)

        label1 = self.lx(label1)
        label2 = self.lx(label2)
        label3 = self.lx(label3)
        
        image = np.concatenate([image1, image2], 0)
        label = np.concatenate([np.expand_dims(label1[0, :, :], axis=0), np.expand_dims(label2[0, :, :], axis=0), np.expand_dims(label3[0, :, :], axis=0)], 0)

        
        return image, label

class cross_entropy(nn.Module):
    def __init__(self):
        super(cross_entropy, self).__init__()
        self.meanloss = luojianet_ms.ops.ReduceMean()

    def call(self, prediction, label):     
        return -self.meanloss(label * luojianet_ms.ops.log(prediction) + (1 - label) * luojianet_ms.ops.log(1 - prediction))



    
def img2ten():
    return luojianet_ms.dataset.py_transforms.Compose([
                c_vision.Resize((256,256)),
                py_vision.ToTensor(),
                c_vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            


if __name__ == '__main__':

    image1_dir = './Building_change_detection/1/A' # A 时相图片路径
    image2_dir = './Building_change_detection/1/B' # B 时相图片路径
    label1_dir = './Building_change_detection/1/building_A' # A 时相建筑物标签路径
    label2_dir = './Building_change_detection/1/building_B' # B 时相建筑物标签路径
    label3_dir = './Building_change_detection/1/OUT' # 变化掩膜路径

    save_dir = './model' # 模型保存路径
    batch_size = 1 
    epochs = 150 # 训练次数
    LR = 1e-4 # 学习率

    changeDatasets = changeDatasets(image1_dir, image2_dir, label1_dir, label2_dir, label3_dir, True)
    Datasets = ds.GeneratorDataset(changeDatasets, ["image", "label"], shuffle=True)
    Datasets = Datasets.batch(batch_size=batch_size)

    net = two_net()
    optimizer = nn.Adam(net.trainable_params(), learning_rate=LR)

    config_ck = CheckpointConfig(save_checkpoint_steps=2, keep_checkpoint_max=10)
    ckpt_cb = ModelCheckpoint(prefix='CD', directory=save_dir, config=config_ck)
    model = luojianet_ms.Model(net, loss_fn=cross_entropy(), optimizer=optimizer)
    model.train(epochs, Datasets, callbacks=[ckpt_cb, LossMonitor(1)])

    

    