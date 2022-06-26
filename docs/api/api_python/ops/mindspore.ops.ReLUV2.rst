mindspore.ops.ReLUV2
====================

.. py:class:: mindspore.ops.ReLUV2()

    线性修正单元激活函数（Rectified Linear Unit activation function）。

    按元素返回 :math:`\max(x,\  0)` 。特别说明，负数输出值会被修改为0，正数输出不受影响。

    .. math::

        \text{ReLU}(x) = (x)^+ = \max(0, x)，

    **输入：**

    - **input_x** (Tensor) - 输入Tensor必须是4-D Tensor。

    **输出：**

    - **output** (Tensor) - 数据类型和shape与 `input_x` 的相同。
    - **mask** (Tensor) - 保留输出，无实际意义。

    **异常：**

    - **TypeError** - `input_x` 不是Tensor。
    - **ValueError** - `input_x` 的shape不是4-D。
