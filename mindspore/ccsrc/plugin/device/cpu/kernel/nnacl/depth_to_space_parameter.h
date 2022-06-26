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
#ifndef MINDSPORE_NNACL_DEPTH_TO_SPACE_PARAMETER_H_
#define MINDSPORE_NNACL_DEPTH_TO_SPACE_PARAMETER_H_
#include "nnacl/op_base.h"

typedef struct DepthToSpaceParameter {
  OpParameter op_parameter_;
  // primitive parameter
  int32_t block_size_;
  // shape correlative
  int32_t in_stride_dim0_;
  int32_t in_stride_dim1_;
  int32_t in_stride_dim2_;
  int32_t out_stride_dim0_;
  int32_t out_stride_dim1_;
  int32_t out_stride_dim2_;
  // other parameter
  uint8_t data_type_size_;
} DepthToSpaceParameter;

#endif  // MINDSPORE_NNACL_DEPTH_TO_SPACE_PARAMETER_H_
