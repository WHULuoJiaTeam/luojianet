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

#ifndef MINDSPORE_NNACL_ARG_MIN_MAX_PARAMETER_H_
#define MINDSPORE_NNACL_ARG_MIN_MAX_PARAMETER_H_

#ifdef ENABLE_ARM64
#include <arm_neon.h>
#endif
#include "nnacl/op_base.h"

typedef int (*COMPARE_FUNCTION)(const void *a, const void *b);

typedef struct ArgElement {
  uint32_t index_;
  union ArgData {
    int8_t i8_data_;
    int32_t i_data_;
    float f_data_;
#ifdef ENABLE_ARM
#if (!SUPPORT_NNIE) || (defined SUPPORT_34XX)
    float16_t f16_data_;
#endif
#endif
  } data_;
} ArgElement;

typedef struct ArgMinMaxParameter {
  OpParameter op_parameter_;
  bool out_value_;
  bool keep_dims_;
  bool get_max_;
  int32_t axis_;
  int32_t topk_;
  int32_t axis_type_;
  int32_t dims_size_;
  int32_t data_type_;  // equals to type_id
  int32_t in_strides_[COMM_SHAPE_SIZE];
  int32_t out_strides_[COMM_SHAPE_SIZE];
  ArgElement *arg_elements_;
} ArgMinMaxParameter;

#endif  // MINDSPORE_NNACL_ARG_MIN_MAX_PARAMETER_H_
