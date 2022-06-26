mindspore.FixedLossScaleManager
===============================

.. py:class:: mindspore.FixedLossScaleManager(loss_scale=128.0, drop_overflow_update=True)

    ��ʧ����ϵ������Ĺ��������̳��� :class:`mindspore.LossScaleManager` ��

    **������**

    - **loss_scale** (float) - �ݶȷŴ�ϵ����ע������� `drop_overflow_update` ��ΪFalse�������Ż���ʱ��Ҫ���Ż����� `loss_scale` ��Ϊ��ͬ��ֵ��Ĭ��ֵ��128.0��
    - **drop_overflow_update** (bool) - �������ʱ���Ƿ�ִ���Ż��������ֵΪTrue����������ʱ����ִ���Ż�����Ĭ��ֵ��True��

    .. py:method:: get_drop_overflow_update()

        ���� `drop_overflow_update` ����ֵ��ʾ�Ƿ��ڷ������ʱ�������ֲ������¡�

        **���أ�**

        bool, `drop_overflow_update` ��ֵ��

    .. py:method:: get_loss_scale()

        ��ȡloss scaleֵ��

        **���أ�**

        bool��`loss_scale` ��ֵ��

    .. py:method:: get_update_cell()

        �������ڸ��� `loss_scale` ֵ�� `Cell` ʵ���� :class:`mindspore.nn.TrainOneStepWithLossScaleCell` ����ø�ʵ��������ʹ�ù̶����ݶȷŴ�ϵ������˸�ʵ����ִ���κβ�����

        **���أ�**

        None�� `Cell` ���� `drop_overflow_update` ΪTrueʱ������ :class:`mindspore.nn.FixedLossScaleUpdateCell` ʵ������ `drop_overflow_update` ΪFalseʱ������None��

    .. py:method:: update_loss_scale(overflow)

        ����loss scaleֵ���� :class:`mindspore.FixedLossScaleManager` �У��÷�����ִ���κβ�����

        **������**

        - **overflow** (bool) - ��ʾ�Ƿ������
