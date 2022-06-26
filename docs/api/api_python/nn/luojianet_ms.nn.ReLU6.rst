mindspore.nn.ReLU6
===================

.. py:class:: mindspore.nn.ReLU6

    ReLU6激活函数。

    ReLU6类似于ReLU，不同之处在于设置了上限，其上限为6，如果输入大于6，输出会被限制为6。公式如下：

    .. math::
        Y = \min(\max(0, x), 6).

    **输入：**
	
    - **x** (Tensor) - ReLU6的输入，任意维度的Tensor，其数据类型为float16或float32。

    **输出：**
	
    Tensor，数据类型与 `x` 相同。

    **异常：**
	
    - **TypeError** - `x` 的数据类型既不是float16也不是float32。