mindspore.ops.Identity
=======================

.. py:class:: mindspore.ops.Identity

    返回与输入具有相同shape和值的Tensor。

    **输入：**

    - **x** (Tensor) - shape为 :math:`(x_1, x_2, ..., x_R)` 的Tensor。数据类型为Number。

    **输出：**

    Tensor，shape和数据类型与输入相同， :math:`(x_1, x_2, ..., x_R)` 。

    **异常：**

    - **TypeError** - `x` 不是Tensor。