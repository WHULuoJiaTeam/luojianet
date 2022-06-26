mindspore.ops.MaxPoolWithArgmax
===============================

.. py:class:: mindspore.ops.MaxPoolWithArgmax(kernel_size=1, strides=1, pad_mode="valid", data_format="NCHW")

    对输入Tensor执行最大池化运算，并返回最大值和索引。

    通常，输入的shape为 :math:`(N_{in}, C_{in}, H_{in}, W_{in})` ，MaxPool在 :math:`(H_{in}, W_{in})` 维度输出区域最大值。 给定 `kernel_size` 为 :math:`(kH, kW)` 和 `stride` ，运算如下：

    .. math::
        \text{output}(N_i, C_j, h, w) = \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1}\\
        \text{input}(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)

    **参数：**

    - **kernel_size** (Union[int, tuple[int]]) - 指定池化核尺寸大小。由一个整数或者是两个整数组成的tuple，表示高和宽。默认值：1。
    - **strides** (Union[int, tuple[int]]) - 池化操作的移动步长，由一个整数或者是两个整数组成的tuple，表示高和宽移动步长。默认值：1。
    - **pad_mode** (str) - 指定池化填充模式，可选值是'same'或'valid'，不区分大小写。默认值：'valid'。

      - **same** - 输出的高度和宽度分别与输入整除 `stride` 后的值相同。
      - **valid** - 在不填充的前提下返回有效计算所得的输出。不满足计算的多余像素会被丢弃。

    - **data_format** (str)：输入和输出的数据格式。可选值为'NHWC'或'NCHW'。默认值：'NCHW'。

    **输入：**

    - **x** (Tensor) - shape为 :math:`(N, C_{in}, H_{in}, W_{in})` 的Tensor。数据类型必须为float16或float32。

    **输出：**

    两个Tensor组成的tuple，表示最大池化结果和生成最大值的位置。

    - **output** (Tensor) - 输出最大池结果，shape为 :math:`(N, C_{out}, H_{out}, W_{out})` 。其数据类型与 `x` 的相同。
    - **mask** (Tensor) - 输出最大值索引。数据类型为int32。

    **异常：**

    - **TypeError** - `x` 的数据类型既不是float16也不是float32。
    - **TypeError** - `kernel_size` 或 `strides` 既不是int也不是tuple。
    - **TypeError** - `x` 不是Tensor。