mindspore.nn.probability.distribution.Categorical
==================================================

.. py:class:: mindspore.nn.probability.distribution.Categorical(probs=None, seed=None, dtype=mstype.int32, name='Categorical')

    分类分布。
    离散随机分布，取值范围为 :math:`\{1, 2, ..., k\}` ，概率质量函数为 :math:`P(X = i) = p_i, i = 1, ..., k`。

    **参数：**

    - **probs** (Tensor，list, numpy.ndarray) - 事件概率。
    - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
    - **dtype** (mindspore.dtype) - 事件样例的类型。默认值：mindspore.int32.
    - **name** (str) - 分布的名称。默认值：Categorical。

    .. note:: 
        `probs` 的秩必须至少为1，值是合适的概率，并且总和为1。

    **异常：**

    - **ValueError** - `probs` 的秩为0或者其中所有元素的和不等于1。

    .. py:method:: probs
        :property:

        返回事件发生的概率。

        **返回：**

        Tensor，事件发生的概率。

    .. py:method:: cdf(value, probs)

        在给定值下计算累积分布函数（Cumulatuve Distribution Function, CDF）。

        **参数：**

        - **value** (Tensor) - 要计算的值。
        - **probs** (Tensor) - 事件发生的概率。默认值：None。

        **返回：**

        Tensor，累积分布函数的值。

    .. py:method:: cross_entropy(dist, probs_b, probs)

        计算分布a和b之间的交叉熵。

        **参数：**

        - **dist** (str) - 分布的类型。
        - **probs_b** (Tensor) - 对比分布的事件发生的概率。
        - **probs** (Tensor) - 事件发生的概率。默认值：None。

        **返回：**

        Tensor，交叉熵的值。

    .. py:method:: entropy(probs)

        计算熵。

        **参数：**

        - **probs_b** (Tensor) - 对比分布的事件发生的概率。默认值：None。

        **返回：**

        Tensor，熵的值。

    .. py:method:: kl_loss(dist, probs)

        计算KL散度，即KL(a||b)。

        **参数：**

        - **dist** (str) - 分布的类型。
        - **probs_b** (Tensor) - 对比分布的事件发生的概率。
        - **probs** (Tensor) - 事件发生的概率。默认值：None。

        **返回：**

        Tensor，KL散度。

    .. py:method:: log_cdf(value, probs)

        计算给定值对于的累积分布函数的对数。

        **参数：**

        - **value** (Tensor) - 要计算的值。
        - **probs** (Tensor) - 事件发生的概率。默认值：None。

        **返回：**

        Tensor，累积分布函数的对数。

    .. py:method:: log_prob(value, probs)

        计算给定值对应的概率的对数。

        **参数：**

        - **value** (Tensor) - 要计算的值。
        - **probs** (Tensor) - 事件发生的概率。默认值：None。

        **返回：**

        Tensor，累积分布函数的对数。

    .. py:method:: log_survival(value, probs)

        计算给定值对应的生存函数的对数。

        **参数：**

        - **value** (Tensor) - 要计算的值。
        - **probs** (Tensor) - 事件发生的概率。默认值：None。

        **返回：**

        Tensor，生存函数的对数。

    .. py:method:: mean(probs)

        计算期望。

        **参数：**

        - **probs** (Tensor) - 事件发生的概率。默认值：None。

        **返回：**

        Tensor，概率分布的期望。

    .. py:method:: mode(probs)

        计算众数。

        **参数：**

        - **probs** (Tensor) - 事件发生的概率。默认值：None。

        **返回：**

        Tensor，概率分布的众数。

    .. py:method:: prob(value, probs)

        计算给定值下的概率。对于离散分布是计算概率质量函数（Probability Mass Function）。

        **参数：**

        - **value** (Tensor) - 要计算的值。
        - **probs** (Tensor) - 事件发生的概率。默认值：None。

        **返回：**

        Tensor，概率值。

    .. py:method:: sample(shape, probs)

        采样函数。

        **参数：**

        - **shape** (tuple) - 样本的shape。
        - **probs** (Tensor) - 事件发生的概率。默认值：None。

        **返回：**

        Tensor，根据概率分布采样的样本。

    .. py:method:: sd(probs)

        计算标准差。

        **参数：**
        
        - **probs** (Tensor) - 事件发生的概率。默认值：None。

        **返回：**

        Tensor，概率分布的标准差。

    .. py:method:: survival_function(value, probs)

        计算给定值对应的生存函数。

        **参数：**

        - **value** (Tensor) - 要计算的值。
        - **probs** (Tensor) - 事件发生的概率。默认值：None。

        **返回：**

        Tensor，生存函数的值。

    .. py:method:: var(probs)

        计算方差。

        **参数：**

        - **probs** (Tensor) - 事件发生的概率。默认值：None。

        **返回：**

        Tensor，概率分布的方差。
