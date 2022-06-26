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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_RESIZE_BILINEAR_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_RESIZE_BILINEAR_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "include/cuda_fp16.h"
template <typename T>
CUDA_LIB_EXPORT void CalResizeBilinear(const T *input, const int n_, const int c_, const int input_h_,
                                       const int input_w_, const int output_h_, const int output_w_,
                                       const float h_scale, const float w_scale, T *output, cudaStream_t cuda_stream);
CUDA_LIB_EXPORT void CalResizeBilinearGrad(const half *input, const int n_, const int c_, const int input_h_,
                                           const int input_w_, const int output_h_, const int output_w_,
                                           const float h_scale, const float w_scale, half *output, float *interim,
                                           cudaStream_t cuda_stream);
CUDA_LIB_EXPORT void CalResizeBilinearGrad(const float *input, const int n_, const int c_, const int input_h_,
                                           const int input_w_, const int output_h_, const int output_w_,
                                           const float h_scale, const float w_scale, float *output, float *interim,
                                           cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_RESIZE_BILINEAR_IMPL_CUH_
