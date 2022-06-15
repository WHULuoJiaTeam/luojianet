# LuojiaNet 在 ModelArts 上的使用（场景分类）

---

## 训练

若要在[LuojiaNet 前端平台](http://58.48.42.237/luojiaNet/)提交训练作业，除了参照[此文件夹](https://github.com/WHULuoJiaTeam/luojianet/edit/master/model_zoo/rs_scene_classification)的示例代码进行结构组织外还需进行以下操作：

1.修改`config.py`中的参数

键值对修改为代码块中所示：

```python
"device_target":"Ascend",      #Ascend
"dataset_path": "/cache/dataset",  #数据存放位置
"save_checkpoint_path": "./checkpoint",  #保存的参数存放位置
```

2.修改`train.py`文件

请分别在文件的不同位置加入如下代码：

**文件头**

```python
import moxing as mox
from luojianet_ms.communication.management import init
```

**文件中**

- 请紧随`if __name__ == '__main__':`插入
  `mox.file.copy_parallel(os.environ['DLS_DATA_URL'], "/cache/dataset")`

- 请在`context.set_context(**kwargs)`或`context.set_auto_parallel_context(**kwargs)`后插入`init()`

  **注意：** 优先插入在`set_auto_parallel_context(**kwargs)`之后

- 请在训练完成后插入
  `mox.file.copy_parallel("./", os.environ['DLS_TRAIN_URL'])`

完整的 Modelarts 用 VGG 示例代码，请点击[此处](https://github.com/WHULuoJiaTeam/luojianet/edit/master/model_zoo/rs_scene_classification/VGG_For_Modelarts)
