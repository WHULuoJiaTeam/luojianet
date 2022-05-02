# 遥感影像变化检测
## 多任务共进的遥感影像建筑物变化检测方法
论文：2022RS《Built-Up Area Change Detection Using Multi-Task Network with Object-Level Refinement》  
链接：[https://doi.org/10.3390/rs14040957](https://doi.org/10.3390/rs14040957)  
代码提供者：[gaosong@whu.edu.cn](gaosong@whu.edu.cn)  
![网络示意图](image.png)
&emsp;
1. 下载遥感建筑物变化检测影像数据集，如WHU ChangeDetection等
2. 将数据集整理成如下格式：  
注意：A、B、building_A、building_B、label中的图片名字必须对应！  
其中，A，building_A分别为A时相影像以及对应的建筑物掩膜，B，building_B分别为B时相影像以及对应的建筑物掩膜，label为对应的变化掩膜

```
.. code-block::
        .
        └── image_folder_dataset_directory
             ├── A
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── B
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── building_A
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── building_B
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── label
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
```
3. 根据需求修改config.py中的参数   
```
    "device_target":"GPU",      #GPU、CPU、Ascend
    "device_id":0,   #显卡ID
    "dataset_path": "./CD_data",  #数据存放位置
    "save_checkpoint_path": "./checkpoint",  #保存的参数存放位置
    "resume":False,   #是否载入模型训练
    "batch_size": 8,
    "aug" : True,
    "step_per_epoch": 200,
    "epoch_size": 200, #训练次数
    "save_checkpoint": True, #是否保存模型
    "save_checkpoint_epochs": 200, #多少次迭代保存一次模型
    "keep_checkpoint_max": 5, #保存模型的最大个数
    "decay_epochs": 20, #学习率衰减的epoch数
    "max_lr": 0.001, #最大学习率
    "min_lr": 0.00001, #最小学习率
    "LR":1e-4
```
4. 设置完毕后，在终端运行``python train.py``进行训练
5. 训练好的模型会根据前面设置的参数保存在相应的目录下，选择合适的模型，使用prediction.py进行推理，在终端运行``python prediction.py --checkpoint_path **** --dataset_path ****``或``python prediction.py --checkpoint_path **** --left_input_file **** --right_input_file ****``进行推理，其参数设置如下   
```
    --checkpoint_path, type=str, default=None, help='Saved checkpoint file path'
    --dataset_path, type=str, default=None, help='Predict dataset path'
    --left_input_file, type=str, default=None, help='Pre-period image'
    --right_input_file, type=str, default=None, help='Post-period image'
    --output_folder, type=str, default="./result", help='Results path'
    --device_target, type=str, default=config.device_target, help='Device target'
    --device_id, type=int, default=config.device_id, help='Device id'
```
6. 训练好的模型会根据前面设置的参数保存在相应的目录下，选择合适的模型，使用eval.py进行评估，在终端运行``python eval.py --checkpoint_path **** --dataset_path ****``进行评估，其参数设置如下   
```
    --checkpoint_path, type=str, default=None, help='Saved checkpoint file path'
    --dataset_path, type=str, default=None, help='Eval dataset path'
    --device_target, type=str, default=config.device_target, help='Device target'
    --device_id, type=int, default=config.device_id, help='Device id'
```