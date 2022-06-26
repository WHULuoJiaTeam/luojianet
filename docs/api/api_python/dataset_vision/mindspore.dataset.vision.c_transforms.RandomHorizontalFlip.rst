mindspore.dataset.vision.c_transforms.RandomHorizontalFlip
==========================================================

.. py:class:: mindspore.dataset.vision.c_transforms.RandomHorizontalFlip(prob=0.5)

    对输入图像按给定的概率进行水平随机翻转。

    .. note:: 此操作支持通过 Offload 在 Ascend 或 GPU 平台上运行。

    **参数：**

    - **prob**  (float, 可选) - 图像被翻转的概率，必须在 [0, 1] 范围内, 默认值：0.5。

    **异常：**

    - **TypeError** - 如果 `prob` 不是float类型。
    - **ValueError** - 如果 `prob` 不在 [0, 1] 范围内。
    - **RuntimeError** - 如果输入图像的shape不是 <H, W> 或 <H, W, C>。
