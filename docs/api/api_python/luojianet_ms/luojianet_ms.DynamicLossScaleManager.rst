mindspore.DynamicLossScaleManager
==================================

.. py:class:: mindspore.DynamicLossScaleManager(init_loss_scale=2 ** 24, scale_factor=2, scale_window=2000)

    ��̬������ʧ����ϵ���Ĺ��������̳��� :class:`mindspore.LossScaleManager` ��

    **������**

    - **init_loss_scale** (float) - ��ʼ�ݶȷŴ�ϵ����Ĭ��ֵ��2**24��
    - **scale_factor** (int) - �Ŵ�/��С������Ĭ��ֵ��2��
    - **scale_window** (int) - �����ʱ����������step�����������Ĭ��ֵ��2000��

    .. py:method:: get_drop_overflow_update()

        ��ֵ��ʾ�Ƿ��ڷ������ʱ�������ֲ������¡�

        **���أ�**

        bool��ʼ��ΪTrue��

    .. py:method:: get_loss_scale()

        ���ص�ǰ�ݶȷŴ�ϵ����

        **���أ�**

        float���ݶȷŴ�ϵ����

    .. py:method:: get_update_cell()

        �������ڸ����ݶȷŴ�ϵ���� `Cell` ʵ����:class:`mindspore.nn.TrainOneStepWithLossScaleCell` ����ø�ʵ����

        **���أ�**

        :class:`mindspore.nn.DynamicLossScaleUpdateCell` ʵ�������ڸ����ݶȷŴ�ϵ����

    .. py:method:: update_loss_scale(overflow)

        �������״̬�����ݶȷŴ�ϵ������������������С�ݶȷŴ�ϵ�������������ݶȷŴ�ϵ����

        **������**

        **overflow** (bool) - ��ʾ�Ƿ������