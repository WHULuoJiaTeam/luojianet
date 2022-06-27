#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import requests
#context
from luojianet_ms import context
from luojianet_ms.communication import init
#dataset transform
import luojianet_ms.dataset as ds
import luojianet_ms.dataset.transforms.c_transforms as C
import luojianet_ms.dataset.vision.c_transforms as CV
from luojianet_ms.dataset.vision import Inter
from luojianet_ms import dtype as mstype
#network
import luojianet_ms.nn as nn
from luojianet_ms.common.initializer import Normal

# training
from luojianet_ms.nn import Accuracy
from luojianet_ms.train.callback import LossMonitor
from luojianet_ms import Model


context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
context.set_auto_parallel_context(parallel_mode=context.ParallelMode.AUTO_PARALLEL)
init("nccl")


requests.packages.urllib3.disable_warnings()
def download_dataset(dataset_url, path):
    filename = dataset_url.split("/")[-1]
    save_path = os.path.join(path, filename)
    if os.path.exists(save_path):
        return
    if not os.path.exists(path):
        os.makedirs(path)
    res = requests.get(dataset_url, stream=True, verify=False)
    with open(save_path, "wb") as f:
        for chunk in res.iter_content(chunk_size=512):
            if chunk:
                f.write(chunk)
    print("The {} file is downloaded and saved in the path {} after processing".format(os.path.basename(dataset_url), path))

train_path = "datasets/MNIST_Data/train"
test_path = "datasets/MNIST_Data/test"
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-labels-idx1-ubyte", train_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-images-idx3-ubyte", train_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-labels-idx1-ubyte", test_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-images-idx3-ubyte", test_path)



def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    # 定义数据集
    mnist_ds = ds.MnistDataset(data_path)
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # 定义所需要操作的map映射
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # 使用map映射函数，将数据操作应用到数据集
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # 进行shuffle、batch、repeat操作
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(count=repeat_size)

    return mnist_ds

class LeNet5(nn.Module):
    """
    Lenet网络结构
    """
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        # 定义所需要的运算
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # 使用定义好的运算构建前向网络
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


net = LeNet5()
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)


# def train_net(model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
#     """定义训练的方法"""
#     # 加载训练数据集
#     ds_train = create_dataset(os.path.join(data_path, "train"), 32, repeat_size)
#     model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(125)], dataset_sink_mode=sink_mode)
def train_net(model, epoch_size, data_path, repeat_size, sink_mode):
    """定义训练的方法"""
    # 加载训练数据集
    ds_train = create_dataset(os.path.join(data_path, "train"), 32, repeat_size)
    model.train(epoch_size, ds_train, dataset_sink_mode=sink_mode)



def test_net(model, data_path):
    """定义验证的方法"""
    ds_eval = create_dataset(os.path.join(data_path, "test"))
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("{}".format(acc))


train_epoch = 1
mnist_path = "./datasets/MNIST_Data"
dataset_size = 1
model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()}, amp_level='O2')
train_net(model, train_epoch, mnist_path, dataset_size, False)
test_net(model, mnist_path)


# # In[ ]:
#
# '''
# 使用以下命令运行脚本：
#
# ```bash
# python lenet.py
# ```
# '''
#
# ## 加载模型
#
#
# # In[13]:
#
#
# from luojianet_ms import load_checkpoint, load_param_into_net
# # 加载已经保存的用于测试的模型
# param_dict = load_checkpoint("checkpoint_lenet-1_1875.ckpt")
# # 加载参数到网络中
# load_param_into_net(net, param_dict)
#
#
# # In[ ]:
#
#
#
# ## 验证模型
# '''
# 我们使用加载的模型和权重进行单个图片数据的分类预测，具体步骤如下：
# '''
#
# # In[ ]:
#
#
# import numpy as np
# from luojianet_ms import Tensor
#
# # 定义测试数据集，batch_size设置为1，则取出一张图片
# ds_test = create_dataset(os.path.join(mnist_path, "test"), batch_size=1).create_dict_iterator()
# data = next(ds_test)
#
# # images为测试图片，labels为测试图片的实际分类
# images = data["image"].asnumpy()
# labels = data["label"].asnumpy()
#
# # 使用函数model.predict预测image对应分类
# output = model.predict(Tensor(data['image']))
# predicted = np.argmax(output.asnumpy(), axis=1)
#
# # 输出预测分类与实际分类
# print(f'Predicted: "{predicted[0]}", Actual: "{labels[0]}"')

