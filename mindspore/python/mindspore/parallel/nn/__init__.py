# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
NOTE:
    Transformer Networks.
    This is an experimental interface that is subject to change or deletion.
    The import path of Transformer APIs have been modified from `mindspore.parallel.nn` to `mindspore.nn.transformer`,
    while the usage of these APIs stay unchanged. The original import path will retain one or two versions.
    You can view the changes using the examples described below:

    # r1.5
    from mindspore.parallel.nn import Transformer

    # Current
    from mindspore.nn.transformer import Transformer
"""
from mindspore import log
from mindspore.nn.transformer import AttentionMask, VocabEmbedding, MultiHeadAttention, FeedForward, \
    TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer, Transformer, \
    TransformerOpParallelConfig, \
    EmbeddingOpParallelConfig, TransformerRecomputeConfig, MoEConfig, FixedSparseAttention, CrossEntropyLoss, \
    OpParallelConfig

__all__ = ["AttentionMask", "VocabEmbedding", "MultiHeadAttention", "FeedForward", "TransformerEncoder",
           "TransformerDecoder", "TransformerEncoderLayer", "TransformerDecoderLayer", "Transformer",
           "TransformerOpParallelConfig", "EmbeddingOpParallelConfig", "TransformerRecomputeConfig", "MoEConfig",
           "FixedSparseAttention", "CrossEntropyLoss", "OpParallelConfig"]

log.warning("'mindspore.parallel.nn' will be deprecated in the future. Please use 'mindspore.nn.transformer' instead.")
