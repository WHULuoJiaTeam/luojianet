# 遥感影像目标检测 Faster-RCNN 网络模型
> Faster-RCNN是Kaiming He等人在2016年提出的端到端的两阶段目标检测算法，也是目前落地最成功的深度学习模型之一，是目标检测领域最经典的模型之一。Faster-RCNN将目标检测任务分成了两个阶段，首先第一阶段，利用深度网络找出图像中共可能存在物体的区域，产生Region Proposal；第二阶段，对Region Proposal内的物体进行分类，并对第一阶段检测的anchor框进行回归。网络损失主要有三部分构成，包括RPN分类、回归损失，以及Bounding Box Head和Classification Head的损失。
>
> 论文：《Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks》
> 链接：https://arxiv.org/abs/1506.01497

> 项目代码参考：https://gitee.com/mindspore/models/tree/master/official/cv/faster_rcnn

<img src="./img/arch.png" style="zoom: 67%;" />



### 一、环境配置

1. 本项目使用的实验环境参考：`requirements.txt`；

   ```shell
   # other packages need to install
   pip install attrdict
   pip install pycocotools
   pip install opencv-python
   conda install pyyaml
   # mmcv
   pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html

### 二、数据集

1. 将数据集组织成以下文件结构，并将标注转化成COCO标注格式（具体可参考[链接](./examples/coco_example.json)）为例，并根据实际需要在`./config/my_config_file.yaml`文件中修改路径配置：

   ```shell
   └── datasets
       ├── train
       │    ├── 000000000001.jpg
       │    ├── 000000000002.jpg
       │    ├── ...
       ├── val
       │    ├── 000000000001.jpg
       │    ├── 000000000002.jpg
       │    ├── ...
       ├── train.json
       ├── val.json
   ```

2. 提供了其他标注文件格式可供选择：若在执行脚本时选择数据集为`other`，需要将数据集信息整理成TXT文件（参考[链接](./examples/txt_example.txt)），每行内容如下：注意需要在`my_config_file.yaml`中将`dataset`参数修改为`other`；

   ```python
   # 数据集目录/文件名 [xmin ymin xmax ymax class] [xmin ymin xmax ymax class]
   train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
   ```

### 三、其他准备

1. 预训练模型：可以从[mindspore hub](https://www.mindspore.cn/hub/docs/zh-CN/master/loading_model_from_hub.html)上进行下载，这里使用的是`imagenet2012`上训练的`resnet152`模型；将预训练模型下载后，放到`./ms_pretrained_model/`目录下；

2. 这里提供了我们在[慧眼“天智杯”人工智能挑战赛](https://www.rsaicp.com/)可见光飞机目标检测数据集上训练好的`faster-rcnn`模型：

   > 链接：https://pan.baidu.com/s/1LGj3N9Jf3GRLZ7SKS93iGQ 
   > 提取码：2022 

### 四、模型训练

1. 下载数据集，并将数据集整理成规定格式；

   > 为了方便测试，这里准备了一个仅包含10张图像的mini_dataset，见`./mini_dataset`；

2. 根据需求修改`config`文件夹下的配置文件，以`example_config.yaml`为例（下面仅列出了主要的待设置参数，其他参数可以配置文件注释）：

   ```python
   # device setting
   device_target: GPU  
   
   # data loading
   img_width: 1024  # 图像尺寸
   img_height: 1024 
   keep_ratio: True  # 是否记录缩放比例
   flip_ratio: 0.5  # 随机水平翻转概率
   expand_ratio: 1.0  # 扩展概率
   
   # LR
   base_lr: 0.02  # 基础学习率
   warmup_step: 500  # warmup的迭代次数
   warmup_ratio: 0.0625 # warmup起止的学习率比例
   
   # train
   batch_size: 4  # 批大小
   momentum: 0.91  # 动量
   weight_decay: 0.00001  # 权重衰减
   epoch_size: 50  # 训练次数
   run_eval: False  # 训练过程中进行测试
   interval: 1  # 测试间隔
   save_checkpoint: True  # 训练过程中供是否储存模型
   save_checkpoint_epochs: 5  # 每隔多少个epoch存储模型
   keep_checkpoint_max: 10  # 存储模型个数上限
   save_checkpoint_path: "./output_dir/resnet152_tianzhi_all/"  # 模型输出路径
   
   # Number of threads used to process the dataset in parallel
   num_parallel_workers: 8  
   # Parallelize Python operations with multiple worker processes
   python_multiprocessing: True
   mindrecord_dir: "./mini_dataset/mindrecord"  # MindRecord文件目录，暂不需要使用
   
   # eval.py and train.py
   coco_root: "./mini_dataset"  # 数据集根目录
   train_data_type: "train"  # 训练集文件夹名称
   val_data_type: "val"  # 验证集文件夹名称
   instance_set: "train.json"  # 标注json文件名称
   coco_classes: ['background', 'ap',]  # 类别名称，主要需要添加background类别
   num_classes: 2  # 类别数目
   prefix: "faster_rcnn.record_"  # MindRecord文件前缀
   
   # train.py FasterRcnn training
   run_distribute: False  # 是否分布式训练
   dataset: "coco"  # 数据集类型: coco or other
   pre_trained: "./ms_pretrained_model/fasterrcnnresnetv1152.ckpt"  # 预训练模型路径
   device_id: 3 
   device_num: 1
   rank_id: 0 
   image_dir: './mini_dataset/train/'  # 图像目录
   anno_path: './mini_dataset/train/train.json'  # 标注文件路径
   backbone: 'resnet_v1_152'  # backbone模型类型

4. 设置完毕后，命令行运行下面的命令进行训练：

   ```shell
   # 单卡训练，是否并行：run_distribute: False   显卡ID：device_id: "0"    显卡数量：device_num: 1
   python train.py --config_path ./config/example_config.yaml
   
   # 多卡训练，需要在config文件中设置相应的参数, 以两卡为例
   # 是否并行：run_distribute: True    显卡ID：device_id: "0,2"    显卡数量：device_num: 2
   mpirun -n 2 python train.py --config_path ./config/example_config.yaml 

5. 训练好的模型会根据config中的参数保存在配置文件中`save_checkpoint_path`参数相应的目录下；损失函数记录在`loss_0.log`文件中；

### 五、模型性能评价

1. 选择合适的模型参数文件，运行`eval.py`进行模型性能评估，测试参数需要在`example_config.yaml`文件中进行配置，或者使用命令行参数进行指定；（目前仅支持`coco`  json格式的标注文件）

   ```shell
   ### example config中进行指定相关参数
   eval_anno_path: "./mini_dataset/train.json"  # 测试集标注文件路径
   eval_checkpoint_path: "./output_dir/resnet152_tianzhi_all/faster_rcnn-500_10.ckpt"  # 加载模型路径
   eval_save_dir: './eval_results/debug'  # 结果存储目录
   
   ### 或者，可以使用命令行参数调用
   python eval.py  
   --config_path path/to/config  # 配置文件路径
   --eval_dataset path/to/dataset  # 数据集路径，对应coco_root
   --annotation path/to/annotation_json  # 标注文件路径，对应val_anno_path
   --result_save_path path/to/save/results  # 输出结果存储路径
   --device CPU/GPU/Ascend   # CPU/GPU/Ascend
   --checkpoint_path path/to/checkpoint # 模型参数路径

### 六、模型推理

1. 选择合适的模型参数文件，运行`inference.py`进行模型推理，测试参数需要在`example_config.yaml`文件中进行配置，或者使用命令行参数进行指定；

   > 样例图片可从下面链接中下载，放在`./examples/inference_images/`目录下。、
   >
   > 链接：https://pan.baidu.com/s/1SVK1EPS0KWPlCD1FpyLckw 
   > 提取码：2022 

   ```shell
   ### example config中进行指定相关参数
   inference_save_dir: './inference_results/'  # 测试结果存储路径：包括图像和txt格式预测结果
   inference_img_dir: './mini_dataset/train/'  # 包含待预测图像的文件夹，注意图像需要是*.png格式
   inference_checkpoint_path: "./output_dir/ckpt_0/faster_rcnn-50_1126.ckpt"  # 模型参数路径
   inference_img_height: 4096  # 待预测图像尺寸
   inference_img_width: 4096
       
   ### 或者，可以使用命令行参数调用
   python inference.py 
   --config_path path/to/config # 配置文件路径 
   --infer_save_dir path/to/save/dir # 推理结果存储文件夹
   --infer_img_dir path/to/inference/img/dir  # 待推理图像文件夹，图像格式*png
   --infer_checkpoint_path path/to/checkpoint  # 模型参数文件夹
   ```

2. 模型推理结果保存在`--infer_save_dir`路径下， 结果包括推理可视化的目标检测结果图，以及检测结果的`result.txt`文件

   - 效果图（飞机目标由红框标出）：

     ![](./img/res_example.png)

   - `txt`文件格式如下：

   ```python
   # txt format: image_name, confidence, [x_min, y_min, x_max, y_max] class_id
   253.png 1.0 1474 2082 1500 2112 1 
   253.png 1.0 1441 2090 1467 2119 1 
   253.png 1.0 1474 2082 1500 2112 1 ....

