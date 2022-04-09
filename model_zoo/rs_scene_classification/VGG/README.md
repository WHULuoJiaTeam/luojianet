# 遥感影像场景训练VGG部分
## 包括VGG11, VGG13, VGG16, VGG19
论文：《Very Deep Convolutional Networks for Large-Scale Image Recognition》
链接：[https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)
1. 下载常用的遥感分类影像数据集，如WHU-RS19、UCMD以及NWPU DataSet等
2. 将数据集整理成如下格式：
```
.. code-block::
    .
    └── image_folder_dataset_directory
            ├── class1
            │    ├── 000000000001.jpg
            │    ├── 000000000002.jpg
            │    ├── ...
            ├── class2
            │    ├── 000000000001.jpg
            │    ├── 000000000002.jpg
            │    ├── ...
            ├── class3
            │    ├── 000000000001.jpg
            │    ├── 000000000002.jpg
            │    ├── ...
            ├── classN
            ├── ...
```
3. 根据需求修改config中的参数
```
    "device_target":"CPU",      #GPU或CPU
    "dataset_path": "WHU-RS19/",  #数据存放位置
    "save_checkpoint_path": "./checkpoint",  #保存的参数存放位置
    "resume":False,   #是否载入模型训练
    "class_num": 19,  #数据集中包含的种类
    "batch_size": 8,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-5,
    "epoch_size": 350, #训练次数
    "save_checkpoint": True, #是否保存模型
    "save_checkpoint_epochs": 1, #多少次迭代保存一次模型
    "keep_checkpoint_max": 5, #文件内保存模型的最多的个数，超过则删掉最原始的
    "opt": 'rmsprop', #优化器：rmsprop或sgd
    "opt_eps": 0.001, 
    "warmup_epochs": 50, #warmup训练策略
    "lr_decay_mode": "warmup", #学习率衰减方式：steps、poly、cosine以及warmup
    "use_label_smooth": True, 
    "label_smooth_factor": 0.1,
    "lr_init": 0.0001, #初始学习率
    "lr_max": 0.1, #最大学习率
    "lr_end": 0.00001 #最小学习率
```
4. 设置完毕后，在cmd下运行``python train.py``进行训练
5. 训练好的模型会根据config中的参数保存在相应的目录下，选择合适的模型，使用eval.py进行测试，在cmd下运行``python eval.py``进行测试验证