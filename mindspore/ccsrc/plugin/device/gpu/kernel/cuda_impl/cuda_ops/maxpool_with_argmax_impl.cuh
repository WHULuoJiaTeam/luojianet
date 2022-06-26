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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MAXPOOL_WITH_ARGMAX_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MAXPOOL_WITH_ARGMAX_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
template <typename T, typename S>
CUDA_LIB_EXPORT void CalMaxPoolWithArgmax(const T* input, const int n, const int c, const int h, const int w,
                                          const int windowHeight, const int windowWidth, const int strideHeight,
                                          const int strideWidth, const int padTop, const int padLeft,
                                          const int outputHeight, const int outputWidth, T* output, S *index,
                                          cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MAXPOOL_WITH_ARGMAX_IMPL_CUH_
