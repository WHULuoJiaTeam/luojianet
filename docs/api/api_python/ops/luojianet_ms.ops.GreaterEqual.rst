mindspore.ops.GreaterEqual
===========================

.. py:class:: mindspore.ops.GreaterEqual

    输入两个数据，逐元素比较第一个数据是否大于等于第二个数据。

    .. note::
        - 输入 `x` 和 `y` 遵循隐式类型转换规则，使数据类型保持一致。
        - 输入必须是两个Tensor，或一个Tensor和一个Scalar。
        - 当输入是两个Tensor时，它们的数据类型不能同时是bool，并保证其shape可以广播。
        - 当输入是一个Tensor和一个Scalar时，Scalar只能是一个常数。
        - 支持广播。
        - 若输入的Tensor可以广播，则会把低维度通过复制该维度的值的方式扩展到另一个输入中对应的高维度。

    .. math::
        out_{i} =\begin{cases}
            & \text{True,    if } x_{i}>=y_{i} \\
            & \text{False,   if } x_{i}<y_{i}
            \end{cases}

    **输入：**

    - **x** (Union[Tensor, Number, bool]) - 第一个输入可以是Number，也可以是数据类型为Number的Tensor。
    - **y** (Union[Tensor, Number, bool]) - 第二个输入是Number，当第一个输入是Tensor时，也可以是bool，或数据类型为Number或bool的Tensor。

    **输出：**

    Tensor，输出的shape与输入广播后的shape相同，数据类型为bool。

    **异常：**

    - **TypeError** - `x` 和 `y` 都不是Tensor。