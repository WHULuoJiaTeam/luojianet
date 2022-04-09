from easydict import EasyDict as ed
'''
You can call the following directory structure from your dataset files and read by LuoJiaNet's API.

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
'''
config = ed({
    "device_target":"GPU",      #GPU或CPU
    "dataset_path": "WHU-RS19/",  #数据存放位置
    "save_checkpoint_path": "./checkpoint",  #保存的参数存放位置
    "resume":False,   #是否载入模型训练
    "class_num": 19,  #数据集中包含的种类
    "batch_size": 8,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-5,
    "epoch_size": 200, #训练次数
    "save_checkpoint": True, #是否保存模型
    "save_checkpoint_epochs": 1, #多少次迭代保存一次模型
    "keep_checkpoint_max": 5, 
    "opt": 'rmsprop', #优化器：rmsprop或sgd
    "opt_eps": 0.001, 
    "warmup_epochs": 50, #warmup训练策略
    "lr_decay_mode": "warmup", #学习率衰减方式：steps、poly、cosine以及warmup
    "use_label_smooth": True, 
    "label_smooth_factor": 0.1,
    "lr_init": 0.0001, #初始学习率
    "lr_max": 0.1, #最大学习率
    "lr_end": 0.00001 #最小学习率
})
