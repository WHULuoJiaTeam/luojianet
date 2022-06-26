mindspore.ops.ReLU
===================

.. py:class:: mindspore.ops.ReLU

    线性修正单元激活函数（Rectified Linear Unit）。

    返回 :math:`\max(x,\  0)` 的值，负值神经元将被设置为0， 正值神经元将保持不变。

    .. math::
        ReLU(x) = (x)^+ = max(0, x)

    .. note::
        一般来说，与 `ReLUV2` 相比，此算子更常用。且 `ReLUV2` 会多输出一个掩码。

    **输入：**

    - **input_x** (Tensor) - ReLU的输入，任意维度的Tensor。其数据类型为数值型。

    **输出：**

    Tensor，数据类型和shape与 `input_x` 相同。

    **异常：**

    - **TypeError** - `input_x` 的数据类型不是数值型。
    - **TypeError** - `input_x` 不是Tensor。