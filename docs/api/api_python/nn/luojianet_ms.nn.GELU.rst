mindspore.nn.GELU
==================

.. py:class:: mindspore.nn.GELU(approximate=True)

    高斯误差线性单元激活函数（Gaussian error linear unit activation function）。

    对输入的每个元素计算GELU。

    GELU的定义如下：

    .. math::
        GELU(x_i) = x_i*P(X < x_i),


    其中 :math:`P` 是标准高斯分布的累积分布函数， :math:`x_i` 是输入的元素。

    GELU相关图参见 `GELU <https://en.wikipedia.org/wiki/Activation_function#/media/File:Activation_gelu.png>`_  。

    **参数：**

    **approximate** (bool): 是否启用approximation，默认值：True。如果approximate的值为True，则高斯误差线性激活函数为:
    
    :math:`0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))` ，

    否则为： :math:`x * P(X <= x) = 0.5 * x * (1 + erf(x / sqrt(2)))`, where P(X) ~ N(0, 1) 。

    **输入：**

    **x** （Tensor） - 用于计算GELU的Tensor。数据类型为float16或float32。shape是 :math:`(N,*)` ， :math:`*` 表示任意的附加维度数。

    **输出：**

    Tensor，具有与 `x` 相同的数据类型和shape。

    **异常：**

    **TypeError** - `x` 的数据类型既不是float16也不是float32。
