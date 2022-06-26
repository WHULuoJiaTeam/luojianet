mindspore.nn.ReLU
=================

.. py:class:: mindspore.nn.ReLU

    修正线性单元激活函数（Rectified Linear Unit activation function）。

    逐元素求 :math:`\max(x,\  0)` 。特别说明，负数输出值会被修改为0，正数输出不受影响。

    .. math::

        \text{ReLU}(x) = (x)^+ = \max(0, x),

    ReLU相关图参见 `ReLU <https://en.wikipedia.org/wiki/Activation_function#/media/File:Activation_rectified_linear.svg>`_ 。

    **输入：**

    - **x** (Tensor) - 用于计算ReLU的任意维度的Tensor。数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore.html#mindspore.dtype>`_。

    **输出：**

    Tensor，数据类型和shape与 `x` 相同。

    **异常：**

    - **TypeError** - `x` 的数据类型不是number。
