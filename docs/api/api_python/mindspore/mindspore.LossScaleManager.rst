mindspore.LossScaleManager
===========================

.. py:class:: mindspore.LossScaleManager

    ʹ�û�Ͼ���ʱ�����ڹ�����ʧ����ϵ����loss scale���ĳ����ࡣ

    ��������Ҫʵ�ָ�������з����� `get_loss_scale` ���ڻ�ȡ��ǰ���ݶȷŴ�ϵ���� `update_loss_scale` ���ڸ����ݶȷŴ�ϵ�����÷�������ѵ�������б����á� `get_update_cell` ���ڻ�ȡ�����ݶȷŴ�ϵ���� `Cell` ʵ������ʵ������ѵ�������б����á���ǰ��ʹ�� `get_update_cell` ��ʽ��

    ���磺:class:`mindspore.FixedLossScaleManager` �� :class:`mindspore.DynamicLossScaleManager` ��

    .. py:method:: get_loss_scale()

        ��ȡ�ݶȷŴ�ϵ����loss scale����ֵ��

    .. py:method:: get_update_cell()

        ��ȡ���ڸ����ݶȷŴ�ϵ����Cellʵ����

    .. py:method:: update_loss_scale(overflow)

        ���� `overflow` ״̬�����ݶȷŴ�ϵ����loss scale)��

        **������**

        - **overflow** (bool) - ��ʾѵ�������Ƿ������