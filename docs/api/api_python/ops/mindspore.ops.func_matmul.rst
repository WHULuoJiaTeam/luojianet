mindspore.ops.matmul
=====================

.. py:function:: mindspore.ops.matmul(x1, x2, dtype=None)

    计算两个数组的乘积。

    .. note::
        不支持NumPy参数 `out` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。在GPU上支持的数据类型为np.float16和np.float32。在CPU上支持的数据类型为np.float16和np.float32。

    **参数：**

    - **x1** (Tensor) - 输入Tensor，不支持Scalar， `x1` 的最后一维度和 `x2` 的倒数第二维度相等，且 `x1` 和 `x2` 彼此支持广播。
    - **x2** (Tensor) - 输入Tensor，不支持Scalar， `x1` 的最后一维度和 `x2` 的倒数第二维度相等，且 `x1` 和 `x2` 彼此支持广播。
    - **dtype** (:class:`mindspore.dtype`, optional) - 指定输入Tensor的数据类型，默认：None。

    **返回：**

    Tensor或Scalar，矩阵乘积的输入。当 `x1` 和 `x2` 为一维向量时，输入为Scalar。

    **异常：**

    - **ValueError** -  `x1` 的最后一维度和 `x2` 的倒数第二维度不相等，或者输入的是Scalar。
    - **ValueError** - `x1` 和 `x2` 彼此不能广播。