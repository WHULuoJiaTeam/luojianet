mindspore.ops.Slice
====================

.. py:class:: mindspore.ops.Slice

    根据指定shape对输入Tensor进行切片。

    从 `begin` 指定的位置开始，根据 `size` 的shape对输入Tensor进行切片。 `begin` 表示 `input_x` 每个维度的偏移量。 `size` 表示输出Tensor的大小。

    .. note::
        `begin` 的起始值为0，`size` 的起始值为1。

    如果 `size[i]` 为-1，则维度i中的所有剩余元素都包含在切片中。这相当于 :math:`size[i] = input_x.shape(i) - begin[i]` 。

    **输入：**

    - **input_x** (Tensor) - Slice的输入，任意维度的Tensor。
    - **begin** (Union[tuple, list]) - 切片的起始位置。只支持常量值（>=0）。
    - **size** (Union[tuple, list]) - 切片的大小。只支持常量值。

    **输出：**

    Tensor，shape与输入 `size` 相同，数据类型与输入 `input_x` 的相同。

    **异常：**

    - **TypeError** - `begin` 或 `size` 既不是tuple也不是list。