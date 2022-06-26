mindspore.dataset.vision.py_transforms.LinearTransformation
===========================================================

.. py:class:: mindspore.dataset.vision.py_transforms.LinearTransformation(transformation_matrix, mean_vector)

    使用指定的变换方阵和均值向量对输入numpy.ndarray图像进行线性变换。

    先将输入图像展平为一维，从中减去均值向量，然后计算其与变换方阵的点积，最后再变形回原始shape。

    **参数：**
        
    - **transformation_matrix** (numpy.ndarray) - shape为(D, D)的变换方阵，其中

      :math:`D = C \times H \times W` 。

    - **mean_vector** (numpy.ndarray) - shape为(D,)的均值向量，其中

      :math:`D = C \times H \times W` 。

    **异常：**

    - **TypeError** - 当 `transformation_matrix` 的类型不为 :class:`numpy.ndarray` 。
    - **TypeError** - 当 `mean_vector` 的类型不为 :class:`numpy.ndarray` 。
