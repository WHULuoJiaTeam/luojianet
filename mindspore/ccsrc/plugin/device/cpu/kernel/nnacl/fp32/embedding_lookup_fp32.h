/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_NNACL_FP32_EMBEDDING_LOOKUP_H_
#define MINDSPORE_NNACL_FP32_EMBEDDING_LOOKUP_H_

#include "nnacl/op_base.h"

typedef struct EmbeddingLookupParameter {
  OpParameter op_parameter_;
  // primitive parameter
  float max_norm_;

  // shape correlative
  bool *is_regulated_;
  int ids_size_;
  int layer_size_;
  int layer_num_;
} EmbeddingLookupParameter;

#ifdef __cplusplus
extern "C" {
#endif
int EmbeddingLookup(float *input_data, const int *ids, float *output_data, const EmbeddingLookupParameter *parameter,
                    int task_id);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_EMBEDDING_LOOKUP_H_
