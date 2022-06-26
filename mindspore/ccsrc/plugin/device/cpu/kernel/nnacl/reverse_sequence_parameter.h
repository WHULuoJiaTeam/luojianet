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

#ifndef MINDSPORE_NNACL_REVERSE_SEQUENCE_PARAMETER_H_
#define MINDSPORE_NNACL_REVERSE_SEQUENCE_PARAMETER_H_

#include "nnacl/op_base.h"

typedef struct ReverseSequenceParameter {
  // primitive parameter
  OpParameter op_parameter_;
  int seq_axis_;
  int batch_axis_;

  // shape correlative
  int input_shape0_[5];
  int output_shape_[5];
  int input_stride_[5];
  int output_stride_[5];

  // other parameter
  int ndim_;
  int outer_count_;
  int outer_stride_;
  int inner_count_;
  int inner_stride_;
  int copy_byte_size_;
  int total_data_size_;
  bool is_seq_length_int32_;
} ReverseSequenceParameter;

#endif  // MINDSPORE_NNACL_REVERSE_SEQUENCE_PARAMETER_H_
