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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BATCHNORM_FOLD2_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BATCHNORM_FOLD2_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
template <typename T>
CUDA_LIB_EXPORT void BatchNormFold2Forward(const T *x, const T *beta, const T *gamma, const T *batch_std,
                                           const T *batch_mean, const T *running_std, const T *running_mean,
                                           const int *global_step, T *y, int freeze_bn, size_t N, size_t C, size_t H,
                                           size_t W, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void CalBatchNormFold2GradNotFreeze(const T *d_beta, const T *reduce_x, const T *batch_mean,
                                                    const T *batch_std, const T *running_mean, const T *running_std,
                                                    const T *gamma, T *d_gamma, T *d_batch_mean, T *d_batch_std,
                                                    size_t C, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void CalBatchNormFold2GradFreeze(const T *d_beta, const T *reduce_x, const T *batch_mean,
                                                 const T *batch_std, const T *running_mean, const T *running_std,
                                                 const T *gamma, T *d_gamma, T *d_batch_mean, T *d_batch_std, size_t C,
                                                 cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void BatchNormFold2GradReduce(const T *dout, const T *x, T *d_beta, T *tmp, T *reduce_x, T *tmp2,
                                              T *tmp_x, size_t N, size_t C, size_t H, size_t W,
                                              cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT void CalBatchNormFold2GradNotFreezeDxMul(const T *batch_std, const T *running_std, T *d_x, size_t N,
                                                         size_t C, size_t H, size_t W, cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BATCHNORM_FOLD2_IMPL_CUH_
