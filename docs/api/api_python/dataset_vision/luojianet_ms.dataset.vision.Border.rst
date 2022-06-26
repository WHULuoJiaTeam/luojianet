mindspore.dataset.vision.Border
===============================

.. py:class:: mindspore.dataset.vision.Border

    边界填充方式枚举类。

    可选枚举值为：Border.CONSTANT、Border.EDGE、Border.REFLECT、Border.SYMMETRIC。

    - **Border.CONSTANT** - 使用常量值进行填充。
    - **Border.EDGE** - 使用各边的边界像素值进行填充。
    - **Border.REFLECT** - 以各边的边界为轴进行镜像填充，忽略边界像素值。
    - **Border.SYMMETRIC** - 以各边的边界为轴进行对称填充，包括边界像素值。

    .. note:: 该类派生自 :class:`str` 以支持 JSON 可序列化。
