﻿mindspore.ops.Sqrt
===================

.. py:class:: mindspore.ops.Sqrt

    计算输入Tensor的平方根。
	
    .. note::
        当输入数据存在一些负数，则负数对应位置上的返回结果为NaN。

    .. math::
        out_{i} =  \sqrt{x_{i}}

    **输入：**

    - **x** (Tensor) - Sqrt的输入，任意维度的Tensor，其秩应小于8，数据类型为数值型。

    **输出：**

    Tensor，shape和数据类型与输入 `x` 相同。

    **异常：**

    - **TypeError** - `x` 不是Tensor。