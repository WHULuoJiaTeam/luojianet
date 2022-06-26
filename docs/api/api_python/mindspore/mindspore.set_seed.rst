mindspore.set_seed
===================

.. py:class:: mindspore.set_seed(seed)

    设置全局种子。

    .. note::
        - 全局种子可用于numpy.random, mindspore.common.Initializer, mindspore.ops.composite.random_ops以及mindspore.nn.probability.distribution。
        - 如果没有设置全局种子，这些包将会各自使用自己的种子，numpy.random和mindspore.common.Initializer将会随机选择种子值，mindspore.ops.composite.random_ops和mindspore.nn.probability.distribution将会使用零作为种子值。
        - numpy.random.seed()设置的种子仅能被numpy.random使用，而这个API设置的种子也可被numpy.random使用，因此推荐使用这个API设置所有的种子。

    **参数：**

    - **seed** (int) – 设置的全局种子。

    **异常:**

    - **ValueError** – 种子值非法 (小于0)。
    - **TypeError** – 种子值非整型数。
