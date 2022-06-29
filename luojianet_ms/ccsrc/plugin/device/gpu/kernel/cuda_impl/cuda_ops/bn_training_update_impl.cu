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

#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/bn_training_update_impl.cuh"

__inline__ __device__ float HalfFloatInputConvert(const half val) { return __half2float(val); }
__inline__ __device__ float HalfFloatInputConvert(const float val) { return val; }

template <typename T>
__device__ __forceinline__ T AbsFunc(T x) {
  return abs(x);
}

template <typename T>
__global__ void BNTrainingUpdateKernel(size_t N, size_t C, size_t H, size_t W, T *x, T *y, float *sum,
                                       float *square_sum, float *scale, float *offset, float *mean,
                                       float *variance, float factor, float epsilon, float *mean_output,
                                       float *variance_output, float *save_mean_reduce_output,
                                       float *save_variance_reduce_output) {
  __shared__ float num_rec;
  int num = N * C * H * W;
  int normal_size = N * H * W;
  num_rec = HalfFloatInputConvert(1) / normal_size;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x) {
    int channel_index = i / (H * W) % C;
    float save_mean_reduce = sum[channel_index] * num_rec;
    float variance_div = square_sum[channel_index] * num_rec;
    float variance_square = save_mean_reduce * save_mean_reduce;
    float save_variance_reduce = variance_div - variance_square;

    float multiplier_add = save_variance_reduce + epsilon;
    if(multiplier_add < static_cast<float>(0)) {
      printf("multiplier_add < 0 %f!\n",multiplier_add);
    }

    float multiplier_sqrt = sqrtf(AbsFunc(multiplier_add));
    float multiplier_div = scale[channel_index] / multiplier_sqrt;

    float addend_mul = multiplier_div * save_mean_reduce;
    float addend_sub = offset[channel_index] - addend_mul;
    T res_y = (multiplier_div * HalfFloatInputConvert(x[i])) + addend_sub;

    float batch_var_scaler;
    if (num == 1) {
      batch_var_scaler = 0.0;
    } else {
      batch_var_scaler = static_cast<float>(num) / (num - 1);
    }
    float batch_variance = save_variance_reduce * batch_var_scaler;

    float factor_reverse = 1.0 - factor;
    float mean_mul = save_mean_reduce * factor;
    float mean_mul_rev = mean[channel_index] * factor_reverse;
    float mean = mean_mul + mean_mul_rev;

    float var_mul = batch_variance * factor;
    float var_mul_rev = variance[channel_index] * factor_reverse;
    float variance = var_mul + var_mul_rev;

    mean_output[channel_index] = mean;
    variance_output[channel_index] = variance;
    save_mean_reduce_output[channel_index] = save_mean_reduce;
    save_variance_reduce_output[channel_index] = save_variance_reduce;
    y[i] = res_y;
  }
  return;
}

template <typename T>
void BNTrainingUpdate(size_t N, size_t C, size_t H, size_t W, T *x, T *y, float *sum, float *square_sum, float *scale,
                      float *offset, float *mean, float *variance, float factor, float epsilon, float *mean_output,
                      float *variance_output, float *save_mean_reduce_output, float *save_variance_reduce_output,
                      cudaStream_t cuda_stream) {
  BNTrainingUpdateKernel<<<C, GET_THREADS, 0, cuda_stream>>>(N, C, H, W, x, y, sum, square_sum, scale, offset, mean,
                                                             variance, factor, epsilon, mean_output, variance_output,
                                                             save_mean_reduce_output, save_variance_reduce_output);
  return;
}

template CUDA_LIB_EXPORT void BNTrainingUpdate<half>(size_t N, size_t C, size_t H, size_t W, half *x, half *y,
                                                     float *sum, float *square_sum, float *scale, float *offset,
                                                     float *mean, float *variance, float factor, float epsilon,
                                                     float *mean_output, float *variance_output,
                                                     float *save_mean_reduce_output, float *save_variance_reduce_output,
                                                     cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void BNTrainingUpdate<float>(size_t N, size_t C, size_t H, size_t W, float *x, float *y,
                                                      float *sum, float *square_sum, float *scale, float *offset,
                                                      float *mean, float *variance, float factor, float epsilon,
                                                      float *mean_output, float *variance_output,
                                                      float *save_mean_reduce_output,
                                                      float *save_variance_reduce_output, cudaStream_t cuda_stream);
