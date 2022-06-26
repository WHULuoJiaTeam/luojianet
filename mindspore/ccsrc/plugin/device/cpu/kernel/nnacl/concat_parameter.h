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

#ifndef MINDSPORE_NNACL_CONCAT_PARAMETER_H_
#define MINDSPORE_NNACL_CONCAT_PARAMETER_H_

#include "nnacl/op_base.h"
#include "nnacl/int8/quantize.h"

typedef struct ConcatParameter {
  OpParameter op_parameter_;
  ConcatQuantArg quant_arg_;
  int axis_;
  int thread_count_;
  int input_num_;
  int **input_shapes_;
  int *output_shapes_;
  int64_t after_axis_size;
  int64_t count_unit_;
} ConcatParameter;

#endif  // MINDSPORE_NNACL_CONCAT_PARAMETER_H_
