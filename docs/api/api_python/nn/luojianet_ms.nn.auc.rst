mindspore.nn.auc
================

.. py:class:: mindspore.nn.auc(x, y, reorder=False)

    使用梯形法则计算曲线下面积AUC（Area Under the Curve，AUC）。这是一个一般函数，给定曲线上的点，
    用于计算ROC (Receiver Operating Curve, ROC) 曲线下的面积。

    **参数：**
    
    - **x** (Union[np.array, list]) - 从ROC曲线（False Positive Rate, FPR）来看，np.array具有假阳性率。如果是多类，则为np.array列表。Shape为 :math:`(N)` 。
    - **y** (Union[np.array, list]) - 从ROC曲线（True Positive Rate, TPR）来看，np.array具有假阳性率。如果是多类，则为np.array列表。Shape为 :math:`(N)` 。

    **返回：**

    float，曲线下面积的值AUC。
