mindspore.dataset.vision.c_transforms.CutOut
============================================

.. py:class:: mindspore.dataset.vision.c_transforms.CutOut(length, num_patches=1)

    从输入图像数组中随机裁剪出给定数量的正方形区域。

    **参数：**

    - **length**  (int) - 每个正方形区域的边长，必须大于 0。
    - **num_patches**  (int, 可选) - 要从图像中切出的正方形区域数，必须大于0，默认值：1。

    **异常：**

    - **TypeError** - 如果 `length` 不是int类型。
    - **TypeError** - 如果 `num_patches` 不是int类型。
    - **ValueError** - 如果 `length` 小于或等于 0。
    - **ValueError** - 如果 `num_patches` 小于或等于 0。
    - **RuntimeError** - 如果输入图像的shape不是 <H, W, C>。
