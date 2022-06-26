mindspore.nn.Vjp
=================

.. py:class:: mindspore.nn.Vjp(fn)

    �����������������ſɱȻ�(vector-Jacobian product, VJP)��VJP��Ӧ `����ģʽ�Զ�΢�� <https://www.mindspore.cn/docs/zh-CN/master/design/gradient.html#id4>`_��

    **������**

    - **fn** (Cell) - ����Cell�����磬���ڽ���Tensor���벢����Tensor����TensorԪ�顣

    **���룺**

    - **inputs** (Tensor) - �����������Σ���������Tensor��
    - **v** (Tensor or Tuple of Tensor) - ���ſɱȾ����˵�������Shape����������һ�¡�

    **�����**

    2��Tensor��TensorԪ�鹹�ɵ�Ԫ�顣

    - **net_output** (Tensor or Tuple of Tensor) - ��������������������
    - **vjp** (Tensor or Tuple of Tensor) - �����ſɱȻ��Ľ����
