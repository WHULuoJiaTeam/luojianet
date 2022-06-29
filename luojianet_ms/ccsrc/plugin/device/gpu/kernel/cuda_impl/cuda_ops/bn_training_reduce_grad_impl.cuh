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

#ifndef LUOJIANET_MS_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BN_TRAINING_REDUCE_GRAD_IMPL_CUH_
#define LUOJIANET_MS_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BN_TRAINING_REDUCE_GRAD_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
template <typename T>
CUDA_LIB_EXPORT void BNTrainingReduceGrad(const T *grads, const float *x, const float *diff_scale,
                                         const float *diff_offset, const float *scale, const float *batch_mean,
                                         const float *batch_variance, T *y, float epsilon, int N, int C, int H,
                                         int W, cudaStream_t cuda_stream);
#endif  // LUOJIANET_MS_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BN_TRAINING_REDUCE_GRAD_IMPL_CUH_
