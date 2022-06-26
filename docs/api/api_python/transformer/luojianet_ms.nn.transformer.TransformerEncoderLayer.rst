﻿.. py:class:: mindspore.nn.transformer.TransformerEncoderLayer(batch_size, hidden_size, ffn_hidden_size, num_heads, seq_length, attention_dropout_rate=0.1, hidden_dropout_rate=0.1, post_layernorm_residual=False, layernorm_compute_type=mstype.float32, softmax_compute_type=mstype.float32, param_init_type=mstype.float32, hidden_act="gelu", use_past=False, moe_config=default_moe_config, parallel_config=default_dpmp_config)

    Transformer的编码器层。Transformer的编码器层上的单层的实现，包括多头注意力层和前馈层。

    **参数：**

    - **batch_size** (int) - 表示输入Tensor的批次大小。
    - **hidden_size** (int) - 表示输入的隐藏大小。
    - **seq_length** (int) - 表示输入序列长度。
    - **ffn_hidden_size** (int) - 表示前馈层中bottleneck的隐藏大小。
    - **num_heads** (int) - 表示注意力头的数量。
    - **hidden_dropout_rate** (float) - 表示作用在隐藏层输出的丢弃率。默认值：0.1
    - **attention_dropout_rate** (float) - 表示注意力score的丢弃率。默认值：0.1
    - **post_layernorm_residual** (bool) - 表示是否在LayerNorm之前使用残差，即是否选择残差为Post-LayerNorm或者Pre-LayerNorm。默认值：False
    - **hidden_act** (str) - 表示内部前馈层的激活函数。其值可为'relu'、'relu6'、'tanh'、'gelu'、'fast_gelu'、'elu'、'sigmoid'、'prelu'、'leakyrelu'、'hswish'、'hsigmoid'、'logsigmoid'等等。默认值：gelu。
    - **layernorm_compute_type** (dtype.Number) - 表示LayerNorm的计算类型。其值应为dtype.float32或dtype.float16。默认值为dtype.float32。
    - **softmax_compute_type** (dtype.Number) - 表示注意力中softmax的计算类型。其值应为dtype.float32或dtype.float16。默认值为mstype.float32。
    - **param_init_type** (dtype.Number) - 表示模块的参数初始化类型。其值应为dtype.float32或dtype.float16。默认值为dtype.float32。
    - **use_past** (bool) - 使用过去状态进行计算，用于增量预测。例如，如果我们有两个单词，想生成十个或以上单词。我们只需要计算一次这两个单词的状态，然后逐个生成下一个单词。当use_past为True时，有两个步骤可以运行预测。第一步是通过 `model.add_flags_recursive(is_first_iteration=True)` 将is_first_iteration设为True，并传递完整的输入。然后，通过 `model.add_flags_recursive(is_first_iteration=False)` 将is_first_iteration设为False。此时，传递step的输入tensor，并对其进行环回。默认值：False
    - **moe_config** (MoEConfig) - 表示MoE (Mixture of Expert)的配置。
    - **parallel_config** (OpParallelConfig) - 表示并行配置。默认值为 `default_dpmp_config` ，表示一个带有默认参数的 `OpParallelConfig` 实例。

    **输入：**

    - **x** (Tensor) - Float Tensor。如果use_past为False或者is_first_iteration为True，shape应为[batch_size, seq_length, hidden_size]或者[batch_size * seq_length, hidden_size]。否则，shape应为[batch_size, 1, hidden_size]。
    - **input_mask** (Tensor) - Float tensor。use_past为False或者is_first_iteration为True时，表示shape为[batch_size, seq_length, seq_length]的注意力掩码。否则，shape应为[batch_size, 1, hidden_size]。
    - **init_reset** (Tensor) - shape为[1]的bool tensor，用于清除增量预测中使用的past key参数和past value参数。仅当use_past为True时有效。默认值为True。
    - **batch_valid_length** (Tensor) - shape为[batch_size]的Int32 tensor，表示过去所计算的索引。当use_past为True时，它用于增量预测。默认值为None。

    **输出：**

    Tuple，表示一个包含(`output`, `layer_present`)的元组。

    - **output** (Tensor) - use_past为False或is_first_iteration为True时，表示shape为(batch_size, seq_length, hidden_size)或(batch_size * seq_length, hidden_size)的层输出的float tensor。否则，shape将为(batch_size, 1, hidden_size)。

    - **layer_present** (Tuple) - 表示shape为((batch_size, num_heads, size_per_head, seq_length)或(batch_size, num_heads, seq_length, size_per_head))的投影key向量和value向量的Tensor的元组。
