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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/fill_impl.cuh"
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void FillKernel(const size_t m, const size_t n, const T *input, T *output) {
  size_t element_num = m * n;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < element_num; pos += blockDim.x * gridDim.x) {
    output[pos] = input[pos % n];
  }
}

template <typename T>
void Fill(const size_t &m, const size_t &n, const T *input, T *output, cudaStream_t stream) {
  FillKernel<<<(m * n + 255) / 256, 256, 0, stream>>>(m, n, input, output);
}

template CUDA_LIB_EXPORT void Fill<float>(const size_t &m, const size_t &n, const float *input, float *output,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT void Fill<half>(const size_t &m, const size_t &n, const half *input, half *output,
                                         cudaStream_t stream);
template CUDA_LIB_EXPORT void Fill<double>(const size_t &m, const size_t &n, const double *input, double *output,
                                           cudaStream_t stream);
