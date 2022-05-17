### 高光谱遥感影像语义分割——FreeNet
![FreeNet网络结构](../../../FreeNet.PNG)

论文：2020TGRS《FPGA: Fast Patch-Free Global Learning Framework for Fully End-to-End Hyperspectral Image Classification》

链接：https://ieeexplore.ieee.org/document/9007624

1.数据集组织格式

将数据集的影像（训练集和测试集的所有影像）放入"image"文件夹，训练集标签放入"train"文件夹，测试集标签放入"test"文件夹，标签的名字和对应影像的名字相同，例如，将数据组织成以下格式：

    └── data
         ├── image
         │    ├── 1.jpg
         │    ├── 2.jpg
         │    ├── 3.jpg
         ├── train
         │    ├── 1.jpg
         │    ├── 2.jpg
         ├── test
         │    ├── 3.jpg
    

2.网络训练

（1）修改config.py中的参数
    
"device_target": "CPU"或"GPU" 
"dataset_path" : 数据集根目录
"normalize": 是否对影像进行归一化，False或True，若为True，则逐波段进行标准差归一化
"nodata_value": 标签中的Nodata值（即不作为样本的像素值）
"in_channels": 输入通道数，即影像波段数
"classnum": 类别数
"batch_size": 训练集batchsize
"num_epochs": 训练迭代次数
"weight": 是否在损失函数中对各类别加权，默认为不加权（None），若需要加权，则给出一个各类别权重的list
"learning_rate": 训练学习率
"save_model_path": 训练模型保存文件夹路径

（2）执行命令

```
python train.py
```
进行网络的训练