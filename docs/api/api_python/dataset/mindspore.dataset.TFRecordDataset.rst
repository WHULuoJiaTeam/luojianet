mindspore.dataset.TFRecordDataset
=================================

.. py:class:: mindspore.dataset.TFRecordDataset(dataset_files, schema=None, columns_list=None, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, shard_equal_rows=False, cache=None)

    读取和解析TFData格式的数据文件构建数据集。生成的数据集的列名和列类型取决于TFRecord文件中的保存的列名与类型。

    **参数：**

    - **dataset_files** (Union[str, list[str]]) - 数据集文件路径，支持单文件路径字符串、多文件路径字符串列表或可被glob库模式匹配的字符串，文件列表将在内部进行字典排序。
    - **schema** (Union[str, Schema], 可选) - 读取模式策略，用于指定读取数据列的数据类型、数据维度等信息。
      支持传入JSON文件路径或 mindspore.dataset.Schema 构造的对象。默认值：None，不指定。
    - **columns_list** (list[str], 可选) - 指定从TFRecord文件中读取的数据列。默认值：None，读取所有列。
    - **num_samples** (int, 可选) - 指定从数据集中读取的样本数，默认值：None，读取全部样本。

      - 如果 `num_samples` 为None，并且numRows字段（由参数 `schema` 定义）不存在，则读取所有数据集；
      - 如果 `num_samples` 为None，并且numRows字段（由参数 `schema` 定义）的值大于0，则读取numRows条数据；
      - 如果 `num_samples` 和numRows字段（由参数 `schema` 定义）的值都大于0，此时仅有参数 `num_samples` 生效且读取给定数量的数据。
    - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用mindspore.dataset.config中配置的线程数。
    - **shuffle** (Union[bool, Shuffle], 可选) - 每个epoch中数据混洗的模式，支持传入bool类型与枚举类型进行指定，默认值：mindspore.dataset.Shuffle.GLOBAL。
      如果 `shuffle` 为False，则不混洗，如果 `shuffle` 为True，等同于将 `shuffle` 设置为mindspore.dataset.Shuffle.GLOBAL。
      通过传入枚举变量设置数据混洗的模式：

      - **Shuffle.GLOBAL**：混洗文件和样本。
      - **Shuffle.FILES**：仅混洗文件。

    - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数，默认值：None。指定此参数后, `num_samples` 表示每个分片的最大样本数。
    - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号，默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
    - **shard_equal_rows** (bool, 可选) - 分布式训练时，为所有分片获取等量的数据行数。默认值：False。如果 `shard_equal_rows` 为False，则可能会使得每个分片的数据条目不相等，从而导致分布式训练失败。因此当每个TFRecord文件的数据数量不相等时，建议将此参数设置为True。注意，只有当指定了 `num_shards` 时才能指定此参数。
    - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/r1.7/dataset/cache.html>`_ 。默认值：None，不使用缓存。

    **异常：**

    - **ValueError** - `dataset_files` 参数所指向的文件无效或不存在。
    - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
    - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
    - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
    - **ValueError** - `shard_id` 参数值错误（小于0或者大于等于 `num_shards` ）。

    .. include:: mindspore.dataset.Dataset.rst

    .. include:: mindspore.dataset.Dataset.b.rst

    .. include:: mindspore.dataset.Dataset.c.rst

    .. include:: mindspore.dataset.Dataset.d.rst

    .. include:: mindspore.dataset.Dataset.zip.rst
