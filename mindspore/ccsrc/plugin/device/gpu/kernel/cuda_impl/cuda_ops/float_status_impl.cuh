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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_FLOAT_STATUS_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_FLOAT_STATUS_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
template <typename T>
CUDA_LIB_EXPORT void CalFloatStatus(const size_t size, const T *input, float *output, cudaStream_t stream);
template <typename T>
CUDA_LIB_EXPORT void CalIsNan(const size_t size, const T *input, bool *output, cudaStream_t stream);
template <typename T>
CUDA_LIB_EXPORT void CalIsInf(const size_t size, const T *input, bool *output, cudaStream_t stream);
template <typename T>
CUDA_LIB_EXPORT void CalIsFinite(const size_t size, const T *input, bool *output, cudaStream_t stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_FLOAT_STATUS_IMPL_CUH_
