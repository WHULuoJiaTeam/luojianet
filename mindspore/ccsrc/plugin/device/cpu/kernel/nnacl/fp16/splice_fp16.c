/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "nnacl/fp16/splice_fp16.h"
void SpliceFp16(const float16_t *src_data, int src_row, int src_col, const SpliceParameter *param, float16_t *dst_data,
                int dst_row, int dst_col) {
  int forward_index = 0;
  for (int r = 0; r < dst_row; ++r) {
    float16_t *dst_row_data = dst_data + r * dst_col;
    for (int off = 0; off < param->context_dim_; ++off) {
      int r_off = param->forward_indexes_[forward_index];
      forward_index++;
      const float16_t *tmp_src_data = src_data + r_off * src_col;
      float16_t *tmp_dst_data = dst_row_data + off * src_col;
      memcpy(tmp_dst_data, tmp_src_data, (size_t)(src_col) * sizeof(float16_t));
    }
  }
}
