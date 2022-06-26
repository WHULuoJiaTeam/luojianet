mindspore.nn.Jvp
=================

.. py:class:: mindspore.nn.Jvp(fn)

    �������������ſɱ�������(Jacobian-vector product, JVP)��JVP��Ӧ `ǰ��ģʽ�Զ�΢�� <https://www.mindspore.cn/docs/zh-CN/master/design/gradient.html#id3>`_��

    **������**

    - **fn** (Cell) - ����Cell�����磬���ڽ���Tensor���벢����Tensor����TensorԪ�顣

    **���룺**

    - **inputs** (Tensor) - �����������Σ���������Tensor��
    - **v** (Tensor or Tuple of Tensor) - ���ſɱȾ����˵�������Shape�����������һ�¡�

    **�����**

    2��Tensor��TensorԪ�鹹�ɵ�Ԫ�顣

    - **net_output** (Tensor or Tuple of Tensor) - ��������������������
    - **jvp** (Tensor or Tuple of Tensor) - �ſɱ��������Ľ����
