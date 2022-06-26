mindspore.nn.Tanh
=============================

.. py:class:: mindspore.nn.Tanh

    Tanh激活函数。

    按元素计算Tanh函数，返回一个新的Tensor，该Tensor是输入元素的双曲正切值。

    Tanh函数定义为：

    .. math::
        tanh(x_i) = \frac{\exp(x_i) - \exp(-x_i)}{\exp(x_i) + \exp(-x_i)} = \frac{\exp(2x_i) - 1}{\exp(2x_i) + 1},

    其中 :math:`x_i` 是输入Tensor的元素。

    **输入：**
    
    - **x** (Tensor) - 任意维度的Tensor，数据类型为float16或float32的输入。

    **输出：**
    
    Tensor，数据类型和shape与 `x` 的相同。

    **异常：**
    
    **TypeError** - `x` 的数据类型既不是float16也不是float32。
