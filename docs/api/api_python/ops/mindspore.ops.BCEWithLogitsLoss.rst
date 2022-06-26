mindspore.ops.BCEWithLogitsLoss
===============================

.. py:class:: mindspore.ops.BCEWithLogitsLoss(reduction='mean')

    输入经过sigmoid激活函数后作为预测值，BCEWithLogitsLoss计算预测值和目标值之间的二值交叉熵损失。

    将输入 `logits` 设置为 :math:`X` ，输入 `labels` 设置为 :math:`Y` ，输入 `weight` 设置为 :math:`W` ，输出设置为 :math:`L` 。则，

    .. math::
        \begin{array}{ll} \\
            L_{ij} = -W_{ij}[Y_{ij}log(X_{ij}) + (1 - Y_{ij})log(1 - X_{ij})]
        \end{array}

    :math:`i` 表示 :math:`i^{th}` 样例， :math:`j` 表示类别。则，

    .. math::
        \ell(x, y) = \begin{cases}
        L, & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`\ell` 表示计算损失的方法。有三种方法：第一种方法是直接提供损失值，第二种方法是计算所有损失的平均值，第三种方法是计算所有损失的总和。

    **参数：**

    - **reduction** (str) - 指定用于输出结果的计算方式。取值为'mean'、'sum'或'none'，不区分大小写。如果'none'，则不执行reduction。默认值：'mean'。

    **输入：**

    - **logits** (Tensor)：输入预测值，任意维度的Tensor。其数据类型为float16或float32。
    - **label** (Tensor)： 输入目标值，shape与 `logits` 相同。数据类型为float16或float32。
    - **weight** (Tensor)：指定每个批次二值交叉熵的权重。支持广播，使其shape与 `logits` 的shape保持一致。数据类型必须为float16或float32。
    - **pos_weight** (Tensor)：指定正类的权重。是一个长度等于分类数的向量。支持广播，使其shape与 `logits` 的shape保持一致。数据类型必须为float16或float32。

    **输出：**

    Tensor或Scalar，如果 `reduction` 为'none'，则为shape和数据类型与输入'logits'相的Tensor。否则，输出为Scalar。

    **异常：**

    - **TypeError** - 任何输入的数据类型既不是float16也不是float32。
    - **ValueError** - `weight` 或 `pos_weight` 不能广播到shape为 `logits` 的Tensor。
    - **ValueError** - `reduction` 不为'none'、'mean'或'sum'。