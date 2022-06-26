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

#include "nnacl/base/broadcast_to.h"
#include <string.h>
#include "nnacl/op_base.h"
#include "nnacl/errorcode.h"

size_t accumulate(const int *shape, int start, int end) {
  size_t product = 1;
  for (int i = start; i <= end; ++i) {
    product *= (size_t)shape[i];
  }
  return product;
}

void pad_input_shape(int *input_shape, int input_shape_len, int output_shape_len) {
  if (input_shape_len < output_shape_len) {
    const int shape_gap = output_shape_len - input_shape_len;
    for (int i = input_shape_len - 1; i >= 0; --i) {
      input_shape[i + shape_gap] = input_shape[i];
    }
    for (int i = 0; i < shape_gap; ++i) {
      input_shape[i] = 1;
    }
  }
}

#define BROADCAST_TO_IMPL(type)                                                                        \
  int broadcast_to_##type(const type *input, BroadcastShapeInfo *shape_info, type *output) {           \
    if (input == NULL || output == NULL) {                                                             \
      return NNACL_NULL_PTR;                                                                           \
    }                                                                                                  \
    if (shape_info->output_shape_size_ > MAX_SHAPE_SIZE) {                                             \
      return NNACL_ERR;                                                                                \
    }                                                                                                  \
    int *input_shape = shape_info->input_shape_;                                                       \
    const int *output_shape = shape_info->output_shape_;                                               \
    const int dim_max = shape_info->output_shape_size_ - 1;                                            \
    const size_t data_length = sizeof(type);                                                           \
    const size_t temp_length = accumulate(output_shape, 0, dim_max);                                   \
    type *data_temp = (type *)malloc(temp_length * data_length);                                       \
    if (data_temp == NULL) {                                                                           \
      return NNACL_ERR;                                                                                \
    }                                                                                                  \
    pad_input_shape(input_shape, shape_info->input_shape_size_, dim_max + 1);                          \
    shape_info->input_shape_size_ = dim_max + 1;                                                       \
                                                                                                       \
    size_t before_dim_elements_num = accumulate(input_shape, 0, dim_max - 1);                          \
    size_t after_dim_elements_num = (size_t)(input_shape[dim_max]);                                    \
    size_t dim_broadcast_rate = (size_t)(output_shape[dim_max] / input_shape[dim_max]);                \
    for (size_t i = 0; i < before_dim_elements_num; ++i) {                                             \
      const type *in_ptr = input + i * after_dim_elements_num;                                         \
      for (size_t j = 0; j < dim_broadcast_rate; ++j) {                                                \
        type *out_ptr = output + (i * dim_broadcast_rate + j) * after_dim_elements_num;                \
        memcpy(out_ptr, in_ptr, after_dim_elements_num *data_length);                                  \
      }                                                                                                \
    }                                                                                                  \
                                                                                                       \
    int dim_index = dim_max - 1;                                                                       \
    while (dim_index >= 0) {                                                                           \
      if (input_shape[dim_index] == 0) {                                                               \
        free(data_temp);                                                                               \
        return NNACL_ERR;                                                                              \
      }                                                                                                \
      dim_broadcast_rate = (size_t)(output_shape[dim_index] / input_shape[dim_index]);                 \
      if (dim_broadcast_rate > 1) {                                                                    \
        before_dim_elements_num = accumulate(input_shape, 0, dim_index - 1);                           \
        after_dim_elements_num = accumulate(output_shape, dim_index + 1, dim_max);                     \
        for (size_t i = 0; i < before_dim_elements_num; ++i) {                                         \
          type *in_ptr = output + i * after_dim_elements_num;                                          \
          for (size_t j = 0; j < dim_broadcast_rate; ++j) {                                            \
            type *out_ptr = data_temp + (i * dim_broadcast_rate + j) * after_dim_elements_num;         \
            memcpy(out_ptr, in_ptr, after_dim_elements_num *data_length);                              \
          }                                                                                            \
        }                                                                                              \
        size_t elements_total = before_dim_elements_num * dim_broadcast_rate * after_dim_elements_num; \
        memcpy(output, data_temp, elements_total *data_length);                                        \
      }                                                                                                \
      --dim_index;                                                                                     \
    }                                                                                                  \
    free(data_temp);                                                                                   \
    return NNACL_OK;                                                                                   \
  }

BROADCAST_TO_IMPL(int)
BROADCAST_TO_IMPL(float)
BROADCAST_TO_IMPL(bool)
#ifdef ENABLE_FP16
BROADCAST_TO_IMPL(float16_t)
#endif
