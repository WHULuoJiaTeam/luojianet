﻿.. py:class:: mindspore.nn.transformer.TransformerDecoder(num_layers, batch_size, hidden_size, ffn_hidden_size, src_seq_length, tgt_seq_length, num_heads, attention_dropout_rate=0.1, hidden_dropout_rate=0.1, post_layernorm_residual=False, layernorm_compute_type=mstype.float32, softmax_compute_type=mstype.float32, param_init_type=mstype.float32, hidden_act="gelu", lambda_func=None, use_past=False, offset=0, moe_config=default_moe_config, parallel_config=default_transformer_config)

    Transformer中的解码器模块，为多层堆叠的 `TransformerDecoderLayer` ，包括多头自注意力层、交叉注意力层和前馈层。

    **参数：**

    - **batch_size** (int) - 表示输入Tensor的批次大小。
    - **num_layers** (int) - 表示 `TransformerDecoderLayer` 的层数。
    - **hidden_size** (int) - 表示输入的隐藏大小。
    - **ffn_hidden_size** (int) - 表示前馈层中bottleneck的隐藏大小。
    - **src_seq_length** (int) - 表示输入源序列长度。
    - **tgt_seq_length** (int) - 表示输入目标序列长度。
    - **num_heads** (int) - 表示注意力头的数量。
    - **hidden_dropout_rate** (float) - 表示作用在隐藏层输出的丢弃率。默认值：0.1
    - **attention_dropout_rate** (float) - 表示注意力score的丢弃率。默认值：0.1
    - **post_layernorm_residual** (bool) - 表示是否在LayerNorm之前使用残差，即是否选择残差为Post-LayerNorm或者Pre-LayerNorm。默认值：False
    - **hidden_act** (str) - 表示内部前馈层的激活函数。其值可为'relu'、'relu6'、'tanh'、'gelu'、'fast_gelu'、'elu'、'sigmoid'、'prelu'、'leakyrelu'、'hswish'、'hsigmoid'、'logsigmoid'等等。默认值：gelu。
    - **layernorm_compute_type** (dtype.Number) - 表示LayerNorm的计算类型。其值应为dtype.float32或dtype.float16。默认值为dtype.float32。
    - **use_past** (bool) - 表示是否开启增量推理。在推理中会缓存注意力机制计算结果，避免冗余计算。默认值为False。
    - **softmax_compute_type** (dtype.Number) - 表示注意力中softmax的计算类型。其值应为dtype.float32或dtype.float16。默认值为mstype.float32。
    - **param_init_type** (dtype.Number) - 表示模块的参数初始化类型。其值应为dtype.float32或dtype.float16。默认值为dtype.float32。
    - **offset** (int) - 表示 `decoder` 的初始层索引偏移值。其用于设置梯度聚合的融合值和流水线并行的stage值，使其不与编码器层的相关属性重叠。
    - **lambda_func** - 表示确定梯度融合索引、pipeline阶段和重计算属性的函数。如果用户想确定pipeline阶段和梯度聚合融合，用户可以传递一个接受 `network` 、 `layer_id` 、 `offset` 、 `parallel_config` 和 `layers` 的函数。 `network(Cell)` 表示transformer块， `layer_id(int)` 表示当前模块的层索引，从零开始计数， `offset(int)` 表示如果网络中还有其他模块，则layer_index需要一个偏置。pipeline的默认设置为： `(layer_id + offset) // (layers / pipeline_stage)` 。默认值：None
    - **moe_config** (MoEConfig) - 表示MoE (Mixture of Expert)的配置。
    - **parallel_config** (TransformerOpParallelConfig) - 表示并行配置。默认值为 `default_transformer_config` ，表示带有默认参数的 `TransformerOpParallelConfig` 实例。

    **输入：**

    - **hidden_stats** (Tensor) - shape为[batch_size, seq_length, hidden_size]或[batch_size * seq_length, hidden_size]的输入tensor。
    - **attention_mask** (Tensor) - shape为[batch_size, seq_length, seq_length]的解码器的注意力掩码。
    - **encoder_output** (Tensor) - shape为[batch_size, seq_length, hidden_size]或[batch_size * seq_length, hidden_size]的编码器的输出。

      .. note::当网络位于最外层时，此参数不能通过None传递。默认值为None。

    - **memory_mask** (Tensor) - shape为[batch, tgt_seq_length, src_seq_length]的交叉注意力的memory掩码，其中tgt_seq_length表示解码器的长度。注：当网络位于最外层时，此参数不能通过None传递。默认值为None。
    - **init_reset** (Tensor) - shape为[1]的bool tensor，用于清除增量预测中使用的past key参数和past value参数。仅当use_past为True时有效。默认值为True。
    - **batch_valid_length** (Tensor) - shape为[batch_size]的Int32 tensor，表示过去所计算的索引。当use_past为True时，它用于增量预测。默认值为None。

    **输出：**

    Tuple，表示一个包含(`output`, `layer_present`)的元组。

    - **output** (Tensor) - 输出的logit。shape为[batch, tgt_seq_length, hidden_size]或[batch * tgt_seq_length, hidden_size]。
    - **layer_present** (Tuple) - 大小为层数的元组，其中每个元组都是shape为((batch_size, num_heads, size_per_head, tgt_seq_length)或(batch_size, num_heads, tgt_seq_length, size_per_head)的自注意力中的投影key向量和value向量的tensor的元组，或者是shape为(batch_size, num_heads, size_per_head, src_seq_length)或(batch_size, num_heads, src_seq_length, size_per_head))的交叉注意力中的投影key向量和value向量的tensor的元组。
