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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/bn_training_update_grad_impl.cuh"

const int kWarpSize = 32;
const int kBlockSize = 1024;
const int kNumWarps = 32;

__inline__ __device__ float HalfFloatInputConvert(const half val) { return __half2float(val); }
__inline__ __device__ float HalfFloatInputConvert(const float val) { return val; }

template <typename T>
__global__ void BNTrainingUpdateGradKernel(size_t N, size_t C, size_t H, size_t W, T *grads, T *x,
                                            float *batch_mean, float *batch_variance, float *diff_scale,
                                            float *diff_offset, float epsilon) {
  int num = N * C * H * W;
  __shared__ T shared_diff_scale[kNumWarps];
  __shared__ T shared_diff_offset[kNumWarps];
  int warpId = threadIdx.x / kWarpSize;
  int laneId = threadIdx.x % kWarpSize;
  int plane = blockIdx.x;
  int plane_size = N * H * W;
  if (threadIdx.x < kNumWarps) {
    shared_diff_scale[threadIdx.x] = static_cast<float>(0);
    shared_diff_offset[threadIdx.x] = static_cast<float>(0);
  }
  __syncthreads();  // ensure all 0 init complete across all values

  float sum_diff_scale = static_cast<float>(0);
  float sum_diff_offset = static_cast<float>(0);


  for (int k = threadIdx.x; k < plane_size; k += blockDim.x) {
    int index = (k / (H * W) * C * H * W) + (plane * H * W) + (k % (H * W));
    int channel_index = index / (H * W) % C;
    float batch_mean_inverse = batch_mean[channel_index] * static_cast<float>(-1);
    float x_sub = batch_mean_inverse + HalfFloatInputConvert(x[index]);
    float data_adds = batch_variance[channel_index] + epsilon;
    float data_rsqrt = sqrtf(data_adds);
    float data_rsqrts = HalfFloatInputConvert(1) / data_rsqrt;
    float x_norm = x_sub * data_rsqrts;
    float input_scale_mul = static_cast<float>(grads[index]) * x_norm;
    float input_grads = HalfFloatInputConvert(grads[index]);
    sum_diff_scale += input_scale_mul;
    sum_diff_offset += input_grads;
  }
  __syncthreads();
  // Warp reduction
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    T other_diff_scale = __shfl_down_sync(0xffffffff, sum_diff_scale, offset);
    T other_diff_offset = __shfl_down_sync(0xffffffff, sum_diff_offset, offset);
    sum_diff_scale += static_cast<float>(other_diff_scale);
    sum_diff_offset += static_cast<float>(other_diff_offset);
  }
  __syncwarp();
  // Move warp-reduction result to shared memory
  if (laneId == 0) {
    shared_diff_scale[warpId] = sum_diff_scale;
    shared_diff_offset[warpId] = sum_diff_offset;
  }
  // Shared memory reduction
  // There are exactly 32 items in shared memory, can be reduced within one warp.
  __syncthreads();
  if (warpId == 0) {
    sum_diff_scale = shared_diff_scale[laneId];
    sum_diff_offset = shared_diff_offset[laneId];
    __syncwarp();
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      T other_diff_scale = __shfl_down_sync(0xffffffff, sum_diff_scale, offset);
      T other_diff_offset = __shfl_down_sync(0xffffffff, sum_diff_offset, offset);
      sum_diff_scale += static_cast<float>(other_diff_scale);
      sum_diff_offset += static_cast<float>(other_diff_offset);
    }
    __syncwarp();
  }
  if (threadIdx.x == 0) {
    diff_scale[plane] = sum_diff_scale;
    diff_offset[plane] = sum_diff_offset;
  }
  return;
}

template <typename T>
void BNTrainingUpdateGrad(size_t N, size_t C, size_t H, size_t W, T *grads, T *x, float *batch_mean,
                          float *batch_variance, float *diff_scale, float *diff_offset, float epsilon,
                          cudaStream_t cuda_stream) {
  BNTrainingUpdateGradKernel<<<C, GET_THREADS, 0, cuda_stream>>>(N, C, H, W, grads, x, batch_mean,
                                                                batch_variance, diff_scale, diff_offset, epsilon);
  return;
}

template CUDA_LIB_EXPORT void BNTrainingUpdateGrad<half>(size_t N, size_t C, size_t H, size_t W, half *grads, half *x,
                                                         float *batch_mean, float *batch_variance, float *diff_scale,
                                                         float *diff_offset, float epsilon, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void BNTrainingUpdateGrad<float>(size_t N, size_t C, size_t H, size_t W, float *grads,
                                                          float *x, float *batch_mean, float *batch_variance,
                                                          float *diff_scale, float *diff_offset, float epsilon,
                                                          cudaStream_t cuda_stream);
