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
#include "bn_training_reduce_impl.cuh"

const int kWarpSize = 32;
const int kNumWarps = 32;

__inline__ __device__ float HalfFloatInputConvert(const half val) { return __half2float(val); }
__inline__ __device__ float HalfFloatInputConvert(const float val) { return val; }


template <typename T>
__global__ void BNTrainingReduceKernel(size_t N, size_t C, size_t H, size_t W, const T *input, T *sum,
                                       T *square_sum) {
  __shared__ T shared_sum[kNumWarps];
  __shared__ T shared_square_sum[kNumWarps];
  int warpId = threadIdx.x / kWarpSize;
  int laneId = threadIdx.x % kWarpSize;

  int plane = blockIdx.x;
  int plane_size = N * H * W;

  if (threadIdx.x < kNumWarps) {
    shared_sum[threadIdx.x] = static_cast<float>(0);
    shared_square_sum[threadIdx.x] = static_cast<float>(0);
  }
  __syncthreads();  // ensure all 0 init complete across all values

  float sum_val = static_cast<float>(0);
  float square_sum_val = static_cast<float>(0);

  for (int x = threadIdx.x; x < plane_size; x += blockDim.x) {
    int index = (x / (H * W) * C * H * W) + (plane * H * W) + (x % (H * W));
    float input_val = HalfFloatInputConvert(input[index]);
    sum_val += input_val;
    square_sum_val += input_val * input_val;
  }
  __syncthreads();

  // Warp reduction
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    float other_sum = __shfl_down_sync(0xffffffff, sum_val, offset);
    float other_square_sum = __shfl_down_sync(0xffffffff, square_sum_val, offset);
    sum_val += other_sum;
    square_sum_val += other_square_sum;
  }
  __syncwarp();
  // Move warp-reduction result to shared memory
  if (laneId == 0) {
    shared_sum[warpId] = sum_val;
    shared_square_sum[warpId] = square_sum_val;
  }

  // Shared memory reduction
  // There are exactly 32 items in shared memory, can be reduced within one warp.
  __syncthreads();
  if (warpId == 0) {
    sum_val = shared_sum[laneId];
    square_sum_val = shared_square_sum[laneId];
    __syncwarp();
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      float other_sum = __shfl_down_sync(0xffffffff, sum_val, offset);
      float other_square_sum = __shfl_down_sync(0xffffffff, square_sum_val, offset);
      sum_val += other_sum;
      square_sum_val += other_square_sum;
    }
    __syncwarp();
  }
  if (threadIdx.x == 0) {
    sum[plane] = static_cast<T>(sum_val);
    square_sum[plane] = static_cast<T>(square_sum_val);
  }
  return;
}

template <typename T>
void BNTrainingReduce(int N, int C, int H, int W, const T *x, T *sum, T *square_sum, cudaStream_t cuda_stream) {
  BNTrainingReduceKernel<<<C, GET_THREADS, 0, cuda_stream>>>(N, C, H, W, x, sum, square_sum);
  return;
}

template CUDA_LIB_EXPORT void BNTrainingReduce<float>(int N, int C, int H, int W, const float *input,
                                                         float *sum, float *square_sum, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BNTrainingReduce<half>(int N, int C, int H, int W, const half *input,
                                                         half *sum, half *square_sum, cudaStream_t cuda_stream);
