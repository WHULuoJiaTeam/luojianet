### 高光谱遥感影像语义分割——FreeNet
![FreeNet网络结构](../../../FreeNet.PNG)

论文：2020TGRS《FPGA: Fast Patch-Free Global Learning Framework for Fully End-to-End Hyperspectral Image Classification》

链接：https://ieeexplore.ieee.org/document/9007624

1.数据集组织格式

将数据集的影像（训练集和测试集的所有影像）放入“image”文件夹，训练集标签放入“train”文件夹，测试集标签放入“test”文件夹，标签的名字和对应影像的名字相同，例如，将数据组织成以下格式：

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
    

