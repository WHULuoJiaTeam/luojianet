mindspore.connect_network_with_dataset
=======================================

.. py:class:: mindspore.connect_network_with_dataset(network, dataset_helper)

    �� `network` �� `dataset_helper` �е����ݼ����ӡ�

    �˺���ʹ�� :class:`mindspore.ops.GetNext` ��װ�������磬�Ա�����������ڼ�����Զ�����������ƶ�Ӧ������ͨ������ȡ���ݣ��������ݴ��ݵ��������硣

    .. note::
        �����ͼģʽ��Ascend/GPU���������磬�˺�����ʹ�� :class:`mindspore.ops.GetNext` ��װ�������硣����������£��������罫��û�иĶ�������·��ء������³�ģʽ�»�ȡ������Ҫʹ�� :class:`mindspore.ops.GetNext` ����˴˺����������ڷ��³�ģʽ��

    **������**

    - **network** (Cell) - ���ݼ���ѵ�����硣
    - **dataset_helper** (DatasetHelper) - һ������MindData���ݼ����࣬�ṩ�����ݼ������͡���״��shape���Ͷ������ƣ��԰�װ :class:`mindspore.ops.GetNext` ��

    **���أ�**

    Cell����Ascend����ͼģʽ�������������£�һ���� :class:`mindspore.ops.GetNext` ��װ�������硣��������������������硣

    **�쳣��**

    - **RuntimeError** - ����ýӿ��ڷ������³�ģʽ���á�
