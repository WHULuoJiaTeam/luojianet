### 高分辨率遥感影像语义分割

模型包含常用的语义分割网络如下：
1.U-Net
论文：《U-Net: Convolutional Networks for Biomedical Image Segmentation》
链接：[https://arxiv.org/abs/1505.04597](http://)

![输入图片说明](UNet.JPG)

1.下载GID 数据集 http://captain.whu.edu.cn/GID/ 数据不裁剪，使用先标记再分块读入的方式，标记json文档格式如下：

```
{
"0": {"imagePath": .....tif", "labelPath":......tif", "x": ..., "y": ..., "block_x": ..., "block_y": ..., "width": ..., "height": ...}, 
"1": {"imagePath": .....tif", "labelPath":......tif", "x": ..., "y": ..., "block_x": ..., "block_y": ..., "width": ..., "height": ...}, 
"2": {"imagePath": .....tif", "labelPath":......tif", "x": ..., "y": ..., "block_x": ..., "block_y": ..., "width": ..., "height": ...},
}
```
imagePath：为文件路径
labelPath：为标签路径
x，y:当前patch左上角坐标
block_x， block_y: 当前patch长宽
width，height：当前影像长宽

2.训练好的模型可使用eval.py进行测试，在cmd下运行python eval.py -d XXX -c XXX -t xxx进行测试验证，也可用python eval.py --dataset_path xxx --checkpoint_path xxx --device_target xxx进行测试验证，输出为验证集的各类别的precision、recall、F1、IoU、mIoU、Kappa精度指标


```
--dataset_path 为验证集路径
--checkpoint_path为训练权重路径
--device_target为设备类型，包括CPU、GPU、Ascend
```

3.利用预训练好的模型对单张影像进行预测，选择合适的模型，使用test.py进行预测，在cmd下运行python prediction.py --input_file ./bridge_1.jpg --output_folder ./output --checkpoint_path xxx.ckpt --device_target xxx，其中,


```
--input_file 为输入的单张影像路径，存储于input_image目录
--output_folder 为输出的结果所在文件夹，存储于output目录，输出结果文件名与输入影像相同，保存为tif格式。例如输入影像名称为bridge_1.tif，则输出名称为bridge_1.tif
--checkpoint_path为训练权重路径，存储于rs_scene_classification_ckp目录
--device_target 为设备类型，包括CPU、GPU、Ascend
```
