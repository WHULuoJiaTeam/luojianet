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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NNACL_FP16_GRAD_RESIZE_GRAD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NNACL_FP16_GRAD_RESIZE_GRAD_H_

#include "nnacl/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ResizeFp16GradParameter {
  OpParameter op_parameter_;
  bool align_corners_;
  int method;
  size_t in_height_;
  size_t in_width_;
  size_t out_height_;
  size_t out_width_;
  float16_t height_scale_;
  float16_t width_scale_;
} ResizeFp16GradParameter;

int ResizeNearestNeighborFp16Grad(float16_t *in_addr, float16_t *out_addr, int batch_size, int channel, int format,
                                  ResizeFp16GradParameter *param);
int ResizeBiLinearFp16Grad(float16_t *in_addr, float16_t *out_addr, int batch_size, int channel, int format,
                           ResizeFp16GradParameter *param);
#ifdef __cplusplus
}
#endif
#endif  //  MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NNACL_FP16_GRAD_RESIZE_GRAD_H_
