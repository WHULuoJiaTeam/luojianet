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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_UNIFORM_CANDIDATE_SAMPLER_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_UNIFORM_CANDIDATE_SAMPLER_IMPL_CUH_
#include <cuda_runtime.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

template <typename S>
CUDA_LIB_EXPORT void CalUniformCandidateSampler(const int64_t true_size, const int64_t num_sampled, const S prob_val,
                                                S *true_expected_count, S *sampled_expected_count,
                                                cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_UNIFORM_CANDIDATE_SAMPLER_IMPL_CUH_
