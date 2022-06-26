mindspore.ops.ReverseV2
========================

.. py:class:: mindspore.ops.ReverseV2(axis)

    对输入Tensor按指定维度反转。

    .. warning::
        "axis"的取值范围为[-dims, dims - 1]，"dims"表示"input_x"的维度长度。

    **参数：**

    - **axis** (Union[tuple(int), list(int)) - 指定反转的轴。

    **输入：**

    - **input_x** (Tensor) - 输入需反转的任意维度的Tensor。数据类型为数值型，不包括float64。

    **输出：**

    Tensor，shape和数据类型与输入 `input_x` 相同。

    **异常：**

    - **TypeError** - `axis` 既不是list也不是tuple。
    - **TypeError** - `axis` 的元素不是int。