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

#include <stdint.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include "bn_training_reduce_grad_impl.cuh"
#include "include/cuda_runtime.h"

__inline__ __device__ float HalfFloatInputConvert(const half val) { return __half2float(val); }
__inline__ __device__ float HalfFloatInputConvert(const float val) { return val; }

template <typename T>
__global__ void BNTrainingReduceGradKernel(const T *grads, const float *x, const float *diff_scale,
                                           const float *diff_offset, const float *scale, const float *batch_mean,
                                           const float *batch_variance, float epsilon, T *y, int N, int C, int H,
                                           int W) {
  __shared__ float num_rec;
  int num = N * C * H * W;
  int normal_size = N * H * W;
  num_rec = HalfFloatInputConvert(1) / normal_size;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x) {
    int channel_index = i / (H * W) % C;
    float data_sqrt = static_cast<float>(sqrt(batch_variance[channel_index] + epsilon));
    float scale_inv = diff_scale[channel_index] * num_rec;
    float offset_inv = diff_offset[channel_index] * num_rec;
    float multiplier = (-1) * scale_inv / data_sqrt;
    float addend_div = batch_mean[channel_index] / data_sqrt;
    float addend_mul = addend_div * scale_inv;
    float addend = addend_mul + offset_inv * (-1);
    float coef_mul = multiplier * x[i];
    float coef_add = HalfFloatInputConvert(grads[i]) + coef_mul;
    float coef = coef_add + addend;
    float mul_scale = scale[channel_index] / data_sqrt;
    y[i] = coef * mul_scale;
  }
  return;
}


template <typename T>
void BNTrainingReduceGrad(const T *grads, const float *x, const float *diff_scale, const float *diff_offset,
                          const float *scale, const float *batch_mean, const float *batch_variance, T *y,
                          float epsilon, int N, int C, int H, int W, cudaStream_t cuda_stream) {
  BNTrainingReduceGradKernel<<<C, GET_THREADS, 0, cuda_stream>>>(grads, x, diff_scale, diff_offset, scale, batch_mean,
                                                                 batch_variance, epsilon, y, N, C, H, W);
}

template CUDA_LIB_EXPORT void BNTrainingReduceGrad<float>(const float *grads, const float *x, const float *diff_scale,
                                                          const float *diff_offset, const float *scale,
                                                          const float *batch_mean, const float *batch_variance,
                                                          float *y, float epsilon,  int N, int C,
                                                          int H, int W, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BNTrainingReduceGrad<half>(const half *grads, const float *x, const float *diff_scale,
                                                          const float *diff_offset, const float *scale,
                                                          const float *batch_mean, const float *batch_variance,
                                                          half *y, float epsilon,  int N, int C,
                                                          int H, int W, cudaStream_t cuda_stream);
