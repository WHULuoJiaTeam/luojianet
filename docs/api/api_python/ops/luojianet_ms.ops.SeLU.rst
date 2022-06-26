mindspore.ops.SeLU
==================

.. py:class:: mindspore.ops.SeLU()

    激活函数SeLU（Scaled exponential Linear Unit）。

    该激活函数定义为：

    .. math::
        E_{i} =
        scale *
        \begin{cases}
        x_{i}, &\text{if } x_{i} \geq 0; \cr
        \text{alpha} * (\exp(x_i) - 1), &\text{otherwise.}
        \end{cases}

    其中， :math:`alpha` 和 :math:`scale` 是预定义的常量（ :math:`alpha=1.67326324` ， :math:`scale=1.05070098` ）。

    更多详细信息，请参见 `Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515>`_ 。

    **输入：**

    - **input_x** (Tensor) - 任意维度的Tensor，数据类型为float16或float32。

    **输出：**

    Tensor，数据类型和shape与 `input_x` 的相同。

    **异常：**

    - **TypeError** - `input_x` 的数据类型既不是float16也不是float32。
