#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 初学入门（训练接口集成版）

'''
`CPU` `GPU` `Linux` `入门`

[![](https://gitee.com/luojianet_ms/docs/raw/master/resource/_static/logo_modelarts.png)](https://authoring-modelarts-cnnorth4.huaweicloud.com/console/lab?share-url-b64=aHR0cHM6Ly9taW5kc3BvcmUtd2Vic2l0ZS5vYnMuY24tbm9ydGgtNC5teWh1YXdlaWNsb3VkLmNvbS9ub3RlYm9vay9tYXN0ZXIvdHV0b3JpYWxzL3poX2NuL21pbmRzcG9yZV9xdWlja19zdGFydC5pcHluYg==&imageid=65f636a0-56cf-49df-b941-7d2a07ba8c8c)&emsp;[![](https://gitee.com/luojianet_ms/docs/raw/master/resource/_static/logo_notebook.png)](https://obs.dualstack.cn-north-4.myhuaweicloud.com/luojianet_ms-website/notebook/master/tutorials/zh_cn/luojianet_ms_quick_start.ipynb)&emsp;[![](https://gitee.com/luojianet_ms/docs/raw/master/resource/_static/logo_download_code.png)](https://obs.dualstack.cn-north-4.myhuaweicloud.com/luojianet_ms-website/notebook/master/tutorials/zh_cn/luojianet_ms_quick_start.py)&emsp;[![](https://gitee.com/luojianet_ms/docs/raw/master/resource/_static/logo_source.png)](https://gitee.com/luojianet_ms/docs/blob/master/tutorials/source_zh_cn/quick_start.ipynb)

本节以Minist数据处理为例，介绍LuoJiaNET搭建LeNet网络、训练和测试的基本流程。

## 配置运行信息

LuoJiaNET通过导入context某块，调用context.set_context`来配置运行需要的信息，如运行设备（CPU/GPU/Ascend），并行计算模式等。
'''

# In[ ]:

import os
import argparse
from luojianet_ms import context

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


# In[ ]:

'''
在上述代码样例中，运行使用图模式，计算设备选用GPU。
'''

# ## 下载数据集
#
# 我们示例中用到的MNIST数据集是由10类28∗28的灰度图片组成，训练数据集包含60000张图片，测试数据集包含10000张图片。
#
# 你可以从[MNIST数据集下载页面](http://yann.lecun.com/exdb/mnist/)下载，并按下方目录结构放置。
#
# 以下示例代码将数据集下载并解压到指定位置。

# In[ ]:


import os
import requests

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


# 下载的数据集文件的目录结构如下：
#
# ```text
# ./datasets/MNIST_Data
# ├── test
# │   ├── t10k-images-idx3-ubyte
# │   └── t10k-labels-idx1-ubyte
# └── train
#     ├── train-images-idx3-ubyte
#     └── train-labels-idx1-ubyte
# ```

# In[ ]:


## 数据处理
'''
数据集对于模型训练非常重要，好的数据集可以有效提高训练精度和效率。
LuoJiaNET融合luojianet_ms特性，提供了用于数据处理的API模块 `luojianet_ms.dataset` , 用于存储样本和标签。
在加载数据集前，我们通常会对数据集进行一些处理，`luojianet_ms.dataset`也集成了常见的数据处理方法。 首先导入LuoJiaNET中`luojianet_ms.dataset`和其他相应的模块。
'''

# In[3]:


import luojianet_ms.dataset as ds
import luojianet_ms.dataset.transforms.c_transforms as C
import luojianet_ms.dataset.vision.c_transforms as CV
from luojianet_ms.dataset.vision import Inter
from luojianet_ms import dtype as mstype


# In[ ]:

'''
通用数据集处理主要分为四个步骤：

1. 定义函数`create_dataset`来创建数据集。
2. 定义需要进行的数据增强和处理操作，为之后进行map映射做准备。
3. 使用map映射函数，将数据操作应用到数据集。
4. 进行数据shuffle、batch操作。
'''

# In[4]:


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


# In[ ]:

'''
其中，`batch_size`为每组包含的数据个数，现设置每组包含32个数据。

## 创建模型

使用LuoJiaNET定义神经网络需要继承`luojianet_ms.nn.Module`,所有算子都继承自Module类

神经网络的各层需要预先在`__init__`方法中定义，然后通过定义`forward`方法来完成神经网络的构造。经典的LeNet的网络结构，定义网络各层如下：
'''

# In[5]:


import luojianet_ms.nn as nn
from luojianet_ms.common.initializer import Normal

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

# 实例化网络
net = LeNet5()


# In[ ]:


## 定义模型损失函数
'''
损失函数是模型迭代训练的目标函数。

LuoJiaNET支持的损失函数有`SoftmaxCrossEntropyWithLogits`、`L1Loss`、`MSELoss`等。这里使用交叉熵损失函数`SoftmaxCrossEntropyWithLogits`。
'''

# In[6]:


# 定义损失函数
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')


# In[ ]:

'''
> 阅读更多有关[在luojianet_ms中使用损失函数](https://www.luojianet_ms.cn/tutorials/zh-CN/master/optimization.html#损失函数)的信息。

LuoJiaNET支持的优化器有`Adam`、`AdamWeightDecay`、`Momentum`等。这里使用`Momentum`优化器为例。
'''

# In[7]:


# 定义优化器
net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)


# In[ ]:

'''
> 阅读更多有关[在luojianet_ms中使用优化器](https://www.luojianet_ms.cn/tutorials/zh-CN/master/optimization.html#优化器)的信息。

## 训练及保存模型

LuoJiaNET框架提供了模型保存的函数`ModelCheckpoint`。
`ModelCheckpoint`可以保存网络模型和参数，以便进行后续的微调操作。
'''

# In[8]:


from luojianet_ms.train.callback import ModelCheckpoint, CheckpointConfig
# 设置模型保存参数
config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
# 应用模型保存参数
ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)


# In[ ]:

'''
LuoJiaNET提供`model.train`接口进行网络训练，`LossMonitor`可监控训练过程中损失值的变化。
'''

# In[9]:


# 导入模型训练需要的库
from luojianet_ms.nn import Accuracy
from luojianet_ms.train.callback import LossMonitor
from luojianet_ms import Model

def train_net(model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    """定义训练的方法"""
    # 加载训练数据集
    ds_train = create_dataset(os.path.join(data_path, "train"), 32, repeat_size)
    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(125)], dataset_sink_mode=sink_mode)


# In[ ]:

'''
通过运行model.eval接口，在训练过程中交叉验证模型训练的精度。
'''

# In[11]:


def test_net(model, data_path):
    """定义验证的方法"""
    ds_eval = create_dataset(os.path.join(data_path, "test"))
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("{}".format(acc))


# 这里把`train_epoch`设置为1，对数据集进行1个迭代的训练。在`train_net`和 `test_net`方法中，我们加载了之前下载的训练数据集，`mnist_path`是MNIST数据集路径。

# In[12]:


train_epoch = 1
mnist_path = "./datasets/MNIST_Data"
dataset_size = 1
model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
train_net(model, train_epoch, mnist_path, dataset_size, ckpoint, False)
test_net(model, mnist_path)


# In[ ]:

'''
使用以下命令运行脚本：

```bash
python lenet.py
```
'''

## 加载模型


# In[13]:


from luojianet_ms import load_checkpoint, load_param_into_net
# 加载已经保存的用于测试的模型
param_dict = load_checkpoint("checkpoint_lenet-1_1875.ckpt")
# 加载参数到网络中
load_param_into_net(net, param_dict)


# In[ ]:



## 验证模型
'''
我们使用加载的模型和权重进行单个图片数据的分类预测，具体步骤如下：
'''

# In[ ]:


import numpy as np
from luojianet_ms import Tensor

# 定义测试数据集，batch_size设置为1，则取出一张图片
ds_test = create_dataset(os.path.join(mnist_path, "test"), batch_size=1).create_dict_iterator()
data = next(ds_test)

# images为测试图片，labels为测试图片的实际分类
images = data["image"].asnumpy()
labels = data["label"].asnumpy()

# 使用函数model.predict预测image对应分类
output = model.predict(Tensor(data['image']))
predicted = np.argmax(output.asnumpy(), axis=1)

# 输出预测分类与实际分类
print(f'Predicted: "{predicted[0]}", Actual: "{labels[0]}"')

