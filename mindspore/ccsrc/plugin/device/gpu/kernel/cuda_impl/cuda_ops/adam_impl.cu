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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adam_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__device__ __forceinline__ T SqrtFunc(T input) {
  return sqrt(input);
}

template <>
__device__ __forceinline__ half SqrtFunc(half input) {
  return hsqrt(input);
}

template <typename T>
__global__ void ApplyAdamKernel(const size_t size, const T *gradient, const T *beta1_power, const T *beta2_power,
                                const T *learning_rate, const T *beta1, const T *beta2, const T *epsilon, T *variable,
                                T *m, T *v) {
  const T one = static_cast<T>(1.0);
  const T new_learning_rate = learning_rate[0] * SqrtFunc(one - beta2_power[0]) / (one - beta1_power[0]);

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    m[i] += (gradient[i] - m[i]) * (one - beta1[0]);
    v[i] += (gradient[i] * gradient[i] - v[i]) * (one - beta2[0]);
    variable[i] -= new_learning_rate * m[i] / (SqrtFunc(v[i]) + epsilon[0]);
  }
}

template <typename T>
__global__ void AdamWeightDecayKernel(const size_t size, const T *gradient, const float *learning_rate,
                                      const float *beta1, const float *beta2, const float *epsilon, const float *decay,
                                      T *variable, T *m, T *v) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    T next_m = beta1[0] * m[i] + (1 - beta1[0]) * gradient[i];
    T next_v = beta2[0] * v[i] + (1 - beta2[0]) * gradient[i] * gradient[i];
    T update = next_m / (sqrt(next_v) + epsilon[0]);
    update += decay[0] * variable[i];
    variable[i] -= learning_rate[0] * update;
    m[i] = next_m;
    v[i] = next_v;
  }
}
template <>
__global__ void AdamWeightDecayKernel(const size_t size, const half *gradient, const float *learning_rate,
                                      const float *beta1, const float *beta2, const float *epsilon, const float *decay,
                                      half *variable, half *m, half *v) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    half next_m = __float2half(beta1[0]) * m[i] + __float2half(1 - beta1[0]) * gradient[i];
    half next_v = __float2half(beta2[0]) * v[i] + __float2half(1 - beta2[0]) * gradient[i] * gradient[i];
    half update = next_m / (hsqrt(next_v) + __float2half(epsilon[0]));
    update += __float2half(decay[0]) * variable[i];
    variable[i] -= __float2half(learning_rate[0]) * update;
    m[i] = next_m;
    v[i] = next_v;
  }
}
template <typename T>
void ApplyAdam(const size_t size, const T *gradient, const T *beta1_power, const T *beta2_power, const T *learning_rate,
               const T *beta1, const T *beta2, const T *epsilon, T *variable, T *m, T *v, cudaStream_t cuda_stream) {
  ApplyAdamKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(
    size, gradient, beta1_power, beta2_power, learning_rate, beta1, beta2, epsilon, variable, m, v);
}
template <typename T>
void AdamWeightDecayOp(const size_t size, const T *gradient, const float *learning_rate, const float *beta1,
                       const float *beta2, const float *epsilon, const float *decay, T *variable, T *m, T *v,
                       cudaStream_t cuda_stream) {
  AdamWeightDecayKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, gradient, learning_rate, beta1, beta2,
                                                                           epsilon, decay, variable, m, v);
}

template CUDA_LIB_EXPORT void ApplyAdam<float>(const size_t size, const float *gradient, const float *beta1_power,
                                               const float *beta2_power, const float *learning_rate, const float *beta1,
                                               const float *beta2, const float *epsilon, float *variable, float *m,
                                               float *v, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ApplyAdam<half>(const size_t size, const half *gradient, const half *beta1_power,
                                              const half *beta2_power, const half *learning_rate, const half *beta1,
                                              const half *beta2, const half *epsilon, half *variable, half *m, half *v,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AdamWeightDecayOp<float>(const size_t size, const float *gradient,
                                                       const float *learning_rate, const float *beta1,
                                                       const float *beta2, const float *epsilon, const float *decay,
                                                       float *variable, float *m, float *v, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AdamWeightDecayOp<half>(const size_t size, const half *gradient,
                                                      const float *learning_rate, const float *beta1,
                                                      const float *beta2, const float *epsilon, const float *decay,
                                                      half *variable, half *m, half *v, cudaStream_t cuda_stream);
