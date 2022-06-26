mindspore.dataset.vision.c_transforms.BoundingBoxAugment
========================================================

.. py:class:: mindspore.dataset.vision.c_transforms.BoundingBoxAugment(transform, ratio=0.3)

    对图像的随机标注边界框区域，应用给定的图像变换处理。

    **参数：**

    - **transform** (TensorOperation) - 对图像的随机标注边界框区域应用的变换处理。
    - **ratio**  (float, 可选) - 要应用变换的边界框的比例。范围：[0.0, 1.0]，默认值：0.3。

    **异常：**

    - **TypeError** - 如果 `transform` 不是 :class:`mindspore.dataset.vision.c_transforms` 模块中的图像变换处理。
    - **TypeError** - 如果 `ratio` 不是float类型。
    - **ValueError** - 如果 `ratio` 不在 [0.0, 1.0] 范围内。
    - **RuntimeError** - 如果给定的边界框无效。
