### 基于大幅面四叉树索引的高分辨率遥感影像语义分割

模型包含常用的语义分割网络如下：
HRNet
论文：《Deep High-Resolution Representation Learning for Visual Recognition》

链接：[https://arxiv.org/abs/1908.07919](http://)

大幅面四叉树方法具体介绍和使用教程可参考
https://luojianet-frontend.obs.cn-central-221.ovaijisuan.com/static/docs/luojianet-docs-r1.7/tutorials/experts/build_zh_cn/html/large_image_processing/binary_tree.html

1. 下载GID数据集
https://captain-whu.github.io/GID/
将数据集整理成如下格式：

```
    |──GID5 			
         |──image(original data)
         |    |──GF2_PMS1__L1A0000647767-MSS1.tif
         |    |──GF2_PMS2__L1A0000635115-MSS2.tif
         |    |──GF2_PMS2__L1A0000788763-MSS2.tif
         |    └─ ...
         └──label(encoded label)
         |    |──GF2_PMS1__L1A0000647767-MSS1_label.tif
         |    |──GF2_PMS2__L1A0000635115-MSS2_label.tif
         |    |──GF2_PMS2__L1A0000788763-MSS2_label.tif    
         |    └─ ...
         └──label_5classes(color label)
         |    |──GF2_PMS1__L1A0000647767-MSS1_label.tif
         |    |──GF2_PMS2__L1A0000635115-MSS2_label.tif
         |    |──GF2_PMS2__L1A0000788763-MSS2_label.tif    
         |    └─ ...
```

有关训练集和验证集，已提供划分好的文件，详见文件夹下的./datalist/train.txt文件和./datalist/valid.txt文件。

2.在train_hrnet_new.py脚本中设置好相应的文件和训练相关参数，即可运行python train_hrnet_new.py进行训练。

```
epochs = 120
initial_learning_rate = 0.0001
eta_min = 1e-5
weight_decay = 1e-5
batch_size_train = 2
batch_size_test = 2
mioubest = 60

image_dir = '/media/xx/PortableSSD/GID5/image/'  # original image
encode_label_dir = '/media/xx/PortableSSD/GID5/label/'  # encoded label
color_label_dir = '/media/xx/PortableSSD/GID5/label_5classes/'  # color label
```

3.训练好的模型可使用eval.py进行测试，运行python eval.py -d XXX -c XXX -t xxx进行测试验证，也可用python eval.py --checkpoint_path xxx --device_target xxx进行测试验证，输出为验证集的各类别的precision、recall、F1、IoU、mIoU、Kappa精度指标

```
--checkpoint_path为训练权重路径
--device_target为设备类型，包括CPU、GPU、Ascend
```

4.利用预训练好的模型对单张影像进行预测，选择合适的模型，使用predict.py进行预测，运行python prediction.py --input_file ./GF2_PMS2__L1A0001799015-MSS2.tif --output_folder ./output --checkpoint_path xxx.ckpt --device_target xxx，其中,

```
--input_file 为输入的单张影像路径，存储于input_image目录
--output_folder 为输出的结果所在文件夹，存储于output目录，输出结果文件名与输入影像相同，保存为tif格式。例如输入影像名称为GF2_PMS2__L1A0001799015-MSS2.tif，则输出名称为GF2_PMS2__L1A0001799015-MSS2.tif
--checkpoint_path为训练权重路径，存储于ckpt目录
--device_target 为设备类型，包括CPU、GPU、Ascend
```
