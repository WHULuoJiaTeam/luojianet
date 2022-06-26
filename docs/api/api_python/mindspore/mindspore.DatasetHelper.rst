mindspore.DatasetHelper
========================

.. py:class:: mindspore.DatasetHelper(dataset, dataset_sink_mode=True, sink_size=-1, epoch_num=1)

    DatasetHelper��һ������MindData���ݼ����࣬�ṩ���ݼ���Ϣ��

    ���ݲ�ͬ�������ģ��ı����ݼ��ĵ������ڲ�ͬ����������ʹ����ͬ�ĵ�����

    .. note::
        DatasetHelper�ĵ������ṩһ��epoch�����ݡ�

    **������**

    - **dataset** (Dataset) - ѵ�����ݼ������������ݼ����������ݼ�������API�� :class:`mindspore.dataset` �����ɣ����� :class:`mindspore.dataset.ImageFolderDataset` ��
    - **dataset_sink_mode** (bool) - ���ֵΪTrue��ʹ�� :class:`mindspore.ops.GetNext` ���豸��Device����ͨ������ͨ���л�ȡ���ݣ�������������Host��ֱ�ӱ������ݼ���ȡ���ݡ�Ĭ��ֵ��True��
    - **sink_size** (int) - ����ÿ���³��е������������ `sink_size` Ϊ-1�����³�ÿ��epoch���������ݼ������ `sink_size` ����0�����³�ÿ��epoch�� `sink_size` ���ݡ�Ĭ��ֵ��-1��
    - **epoch_num** (int) - ���ƴ����͵�epoch��������Ĭ��ֵ��1��

    .. py:method:: continue_send()
        
        ��epoch��ʼʱ�������豸�������ݡ�

    .. py:method:: dynamic_min_max_shapes()
        
        ���ض�̬���ݵ���״(shape)��Χ����С��״(shape)�������״(shape)����

    .. py:method:: get_data_info()
        
        �³�ģʽ�£���ȡ��ǰ�������ݵ����ͺ���״(shape)��ͨ����������״(shape)��̬�仯�ĳ���ʹ�á�

    .. py:method:: release()
        
        �ͷ������³���Դ��

    .. py:method:: sink_size()
        
        ��ȡÿ�ε����� `sink_size` ��

    .. py:method:: stop_send()
        
        ֹͣ���������³����ݡ�

    .. py:method:: types_shapes()
        
        �ӵ�ǰ�����е����ݼ���ȡ���ͺ���״(shape)��
