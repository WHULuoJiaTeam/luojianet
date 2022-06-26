mindspore.dataset.transforms.py_transforms.RandomOrder
======================================================

.. py:class:: mindspore.dataset.transforms.py_transforms.RandomOrder(transforms)

    给一个数据增强的列表，随机打乱数据增强处理的顺序。

    **参数：**

    - **transforms** (list) - 一个数据增强的列表。

    **异常：**
      
    - **TypeError** - 参数 `transforms` 类型不为list。
    - **TypeError** - 参数 `transforms` 的元素不是Python可调用对象或py_transforms模块中的数据处理操作。
    - **ValueError** - 参数 `transforms` 是空的list。
