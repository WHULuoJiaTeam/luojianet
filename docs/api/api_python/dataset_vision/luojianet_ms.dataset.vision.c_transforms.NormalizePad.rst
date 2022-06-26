mindspore.dataset.vision.c_transforms.NormalizePad
==================================================

.. py:class:: mindspore.dataset.vision.c_transforms.NormalizePad(mean, std, dtype="float32")

    根据均值和标准差对输入图像进行归一化，然后填充一个全零的额外通道。

    **参数：**

    - **mean**  (sequence) - 图像每个通道的均值组成的列表或元组。平均值必须在 (0.0, 255.0] 范围内。
    - **std**  (sequence) - 图像每个通道的标准差组成的列表或元组。标准差值必须在 (0.0, 255.0] 范围内。
    - **dtype**  (str, 可选) - 输出图像的数据类型，默认值："float32"。

    **异常：**

    - **TypeError** - 如果 `mean` 不是sequence类型。
    - **TypeError** - 如果 `std` 不是sequence类型。
    - **TypeError** - 如果 `dtype` 不是str类型。
    - **ValueError** - 如果 `mean` 不在 [0.0, 255.0] 范围内。
    - **ValueError** - 如果 `std` 不在范围内 (0.0, 255.0]。
    - **RuntimeError** - 如果输入图像的shape不是 <H, W, C>。
