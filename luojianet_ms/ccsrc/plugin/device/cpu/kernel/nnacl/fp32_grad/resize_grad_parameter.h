/**
 * Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022 Huawei Technologies Co., Ltd
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

#ifndef LUOJIANET_MS_NNACL_FP32_GRAD_RESIZE_PARAMETER_GRAD_H_
#define LUOJIANET_MS_NNACL_FP32_GRAD_RESIZE_PARAMETER_GRAD_H_

#include "nnacl/op_base.h"

typedef struct ResizeGradParameter {
  OpParameter op_parameter_;
  bool align_corners_;
  int method;
  size_t in_height_;
  size_t in_width_;
  size_t out_height_;
  size_t out_width_;
  float height_scale_;
  float width_scale_;
} ResizeGradParameter;

#endif  //  LUOJIANET_MS_NNACL_FP32_GRAD_RESIZE_PARAMETER_GRAD_H_
