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

#ifndef MINDSPORE_NNACL_SIGMOID_PARAMETER_H_
#define MINDSPORE_NNACL_SIGMOID_PARAMETER_H_

#include "nnacl/op_base.h"

typedef struct SigmoidParameter {
  // primitive parameter
  OpParameter op_parameter_;

  // shape correlative
  const int *in_shape_;
  const int *out_shape_;

  // other parameter
  SigmoidQuantArg quant_arg;
  double alpha_;
  int thread_count_;
  int64_t offset_[MAX_SHAPE_SIZE];
  int64_t in_offset_[MAX_SHAPE_SIZE];
  int64_t axis_;
  int input_dim_;
  int element_num;
} SigmoidParameter;

#endif  // MINDSPORE_NNACL_SIGMOID_PARAMETER_H_
