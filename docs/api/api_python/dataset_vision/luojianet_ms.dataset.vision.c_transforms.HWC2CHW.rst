mindspore.dataset.vision.c_transforms.HWC2CHW
=============================================

.. py:class:: mindspore.dataset.vision.c_transforms.HWC2CHW()

    将输入图像的shape从 <H, W, C> 转换为 <C, H, W>。输入图像应为 3 通道图像。

    .. note:: 此操作支持通过 Offload 在 Ascend 或 GPU 平台上运行。

    **异常：**

    - **RuntimeError** - 如果输入图像的shape不是 <H, W, C>。
