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

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/layer_norm_grad_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/layer_norm_impl.cuh"
#include "include/cuda_fp16.h"

constexpr int NUM_PER_THREAD_REDUCE = 4;
constexpr int WARP_SIZE = 32;

template <typename T>
inline __device__ T my_pow(T a, double b) {
  return pow(a, static_cast<float>(b));
}

template <>
inline __device__ half my_pow(half a, double b) {
  return __float2half(pow(__half2float(a), static_cast<float>(b)));
}

template <typename T>
inline __device__ void GammaAndBetaThreadReduce(const int &col, const int &row_dim, const int &col_dim,
                                                const int &mean_dim, const T &epsilon, const T *dy, const T *x,
                                                const T *mean, const T *var, T *dg, T *db) {
  int loop_num = (row_dim + NUM_PER_THREAD_REDUCE - 1) / NUM_PER_THREAD_REDUCE;
  for (int i = threadIdx.x; i < loop_num; i += blockDim.x) {
    for (int j = 0; j < NUM_PER_THREAD_REDUCE; j++) {
      int row = NUM_PER_THREAD_REDUCE * i + j;
      if (row >= row_dim) {
        return;
      }

      int pos = row * col_dim + col;
      int mean_offset = pos / mean_dim;
      dg[0] += dy[pos] * my_pow(var[mean_offset] + epsilon, -0.5) * (x[pos] - mean[mean_offset]);
      db[0] += dy[pos];
    }
  }
}

template <typename T>
inline __device__ void GammaAndBetaWarpReduce(T *dg, T *db) {
  for (int delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
    dg[0] += __shfl_down_sync(0xffffffff, dg[0], delta);
    db[0] += __shfl_down_sync(0xffffffff, db[0], delta);
  }
}

template <typename T>
inline __device__ void GammaAndBetaBlockReduce(const int &col, const int &row_dim, T *dg, T *db, T *dg_addr,
                                               T *db_addr) {
  // load data to share memory
  // thread(0, 32, 64, 96, ...) keep the data
  DynamicSharedMem<T> share_mem;
  if (threadIdx.x % WARP_SIZE == 0) {
    int offset = threadIdx.x / WARP_SIZE * 2;
    share_mem.addr()[offset] = dg[0];
    share_mem.addr()[offset + 1] = db[0];
  }
  __syncthreads();

  for (int stride = blockDim.x / WARP_SIZE / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      int offset = (threadIdx.x + stride) * 2;
      share_mem.addr()[threadIdx.x * 2] += share_mem.addr()[offset];
      share_mem.addr()[threadIdx.x * 2 + 1] += share_mem.addr()[offset + 1];
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    dg_addr[col] = share_mem.addr()[0];
    db_addr[col] = share_mem.addr()[1];
  }
}

template <typename T>
__global__ void GammaAndBetaPropKernel(const int row_dim, const int col_dim, const int mean_dim, const T epsilon,
                                       const T *dy, const T *x, const T *mean_addr, const T *var_addr, T *dg_addr,
                                       T *db_addr) {
  // row: [0:param_axis]
  // col: [param_axis:]
  // dg[i][j] = dy[i][j] * (var[i] + epsilon, -0.5) * (x[i][j] - mean[i])
  // dg[j] = \Sigma_{j}dg[i][j]
  for (int col = blockIdx.x; col < col_dim; col += gridDim.x) {
    T dg = 0;
    T db = 0;
    GammaAndBetaThreadReduce(col, row_dim, col_dim, mean_dim, epsilon, dy, x, mean_addr, var_addr, &dg, &db);
    GammaAndBetaWarpReduce(&dg, &db);
    GammaAndBetaBlockReduce(col, row_dim, &dg, &db, dg_addr, db_addr);
  }
}

template <typename T>
inline __device__ void InputThreadReduce(const int &row, const int &col_dim, const int &param_dim, const T &epsilon,
                                         T *sum1, T *sum2, T *sum3, const T *dy, const T *x, const T *mean,
                                         const T *var, const T *gamma) {
  int loop_num = (col_dim + NUM_PER_THREAD_REDUCE - 1) / NUM_PER_THREAD_REDUCE;
  for (int i = threadIdx.x; i < loop_num; i += blockDim.x) {
    for (int j = 0; j < NUM_PER_THREAD_REDUCE; j++) {
      int col = NUM_PER_THREAD_REDUCE * i + j;
      if (col >= col_dim) {
        return;
      }

      int pos = row * col_dim + col;
      int gamma_offset = pos % param_dim;
      T v1 = dy[pos] * gamma[gamma_offset];
      T v2 = x[pos] - mean[row];

      sum1[0] += -0.5 * v1 * v2 * my_pow(var[row] + epsilon, -1.5);
      sum2[0] += v1;
      sum3[0] += -2.0 * v2;
    }
  }
}

template <>
inline __device__ void InputThreadReduce(const int &row, const int &col_dim, const int &param_dim, const half &epsilon,
                                         half *sum1, half *sum2, half *sum3, const half *dy, const half *x,
                                         const half *mean, const half *var, const half *gamma) {
  int loop_num = (col_dim + NUM_PER_THREAD_REDUCE - 1) / NUM_PER_THREAD_REDUCE;
  for (int i = threadIdx.x; i < loop_num; i += blockDim.x) {
    for (int j = 0; j < NUM_PER_THREAD_REDUCE; j++) {
      int col = NUM_PER_THREAD_REDUCE * i + j;
      if (col >= col_dim) {
        return;
      }

      int pos = row * col_dim + col;
      int gamma_offset = pos % param_dim;
      half v1 = dy[pos] * gamma[gamma_offset];
      half v2 = x[pos] - mean[row];

      sum1[0] += __float2half(-0.5) * v1 * v2 * my_pow(var[row] + epsilon, -1.5);
      sum2[0] += v1;
      sum3[0] += __float2half(-2.0) * v2;
    }
  }
}

template <typename T>
inline __device__ void InputWarpReduce(T *sum1, T *sum2, T *sum3) {
  for (int delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
    sum1[0] += __shfl_down_sync(0xffffffff, sum1[0], delta);
    sum2[0] += __shfl_down_sync(0xffffffff, sum2[0], delta);
    sum3[0] += __shfl_down_sync(0xffffffff, sum3[0], delta);
  }
}

template <typename T>
inline __device__ void InputBlockReduce(const int &col_dim, T *sum1, T *sum2, T *sum3, T *share_mem) {
  // load data to share memory
  // thread(0, 32, 64, 96, ...) keep the data
  if (threadIdx.x % WARP_SIZE == 0) {
    int offset = threadIdx.x / WARP_SIZE * 3;
    share_mem[offset] = sum1[0];
    share_mem[offset + 1] = sum2[0];
    share_mem[offset + 2] = sum3[0];
  }
  __syncthreads();

  for (int stride = blockDim.x / WARP_SIZE / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      int offset = (threadIdx.x + stride) * 3;
      share_mem[threadIdx.x * 3] += share_mem[offset];
      share_mem[threadIdx.x * 3 + 1] += share_mem[offset + 1];
      share_mem[threadIdx.x * 3 + 2] += share_mem[offset + 2];
    }
  }
  __syncthreads();
}

template <typename T>
inline __device__ void InputProp(const int &row, const int &col_dim, const int &param_dim, const T &epsilon,
                                 const T *dy, const T *x, const T *mean, const T *var, const T *gamma, T *dx,
                                 const T *share_mem) {
  for (int col = threadIdx.x; col < col_dim; col += blockDim.x) {
    int pos = (row * col_dim + col);
    int gamma_offset = pos % param_dim;
    T v1 = dy[pos] * gamma[gamma_offset];
    T v2 = x[pos] - mean[row];
    T v3 = my_pow(var[row] + epsilon, -0.5);
    dx[pos] = v1 * v3 + share_mem[0] * (2.0 / col_dim) * v2 +
              (-1.0 * v3 * share_mem[1] + (1.0 / col_dim) * share_mem[0] * share_mem[2]) * (1.0 / col_dim);
  }
}

template <>
inline __device__ void InputProp(const int &row, const int &col_dim, const int &param_dim, const half &epsilon,
                                 const half *dy, const half *x, const half *mean, const half *var, const half *gamma,
                                 half *dx, const half *share_mem) {
  for (int col = threadIdx.x; col < col_dim; col += blockDim.x) {
    int pos = (row * col_dim + col);
    int gamma_offset = pos % param_dim;
    half v1 = dy[pos] * gamma[gamma_offset];
    half v2 = x[pos] - mean[row];
    half v3 = my_pow(var[row] + epsilon, -0.5);
    dx[pos] = v1 * v3 + share_mem[0] * __float2half(2.0 / col_dim) * v2 +
              (__float2half(-1.0) * v3 * share_mem[1] + __float2half(1.0 / col_dim) * share_mem[0] * share_mem[2]) *
                __float2half(1.0 / col_dim);
  }
}

template <typename T>
__global__ void InputPropKernel(const int row_dim, const int col_dim, const int param_dim, const T epsilon, const T *dy,
                                const T *x, const T *mean, const T *var, const T *gamma, T *dx) {
  for (int row = blockIdx.x; row < row_dim; row += gridDim.x) {
    T sum1 = 0;
    T sum2 = 0;
    T sum3 = 0;
    DynamicSharedMem<T> share_mem;
    InputThreadReduce(row, col_dim, param_dim, epsilon, &sum1, &sum2, &sum3, dy, x, mean, var, gamma);
    InputWarpReduce(&sum1, &sum2, &sum3);
    InputBlockReduce(col_dim, &sum1, &sum2, &sum3, share_mem.addr());
    InputProp(row, col_dim, param_dim, epsilon, dy, x, mean, var, gamma, dx, share_mem.addr());
  }
}

template <typename T>
void LayerNormGrad(const int &row_dim, const int &col_dim, const int &param_dim, const T &epsilon, const T *dy,
                   const T *x, const T *mean, const T *var, const T *gamma, T *dx, T *dg, T *db, cudaStream_t stream) {
  const int thread_per_block = 256;
  int share_mem_size = thread_per_block / WARP_SIZE * 3 * sizeof(T);
  InputPropKernel<<<row_dim, thread_per_block, share_mem_size, stream>>>(row_dim, col_dim, param_dim, epsilon, dy, x,
                                                                         mean, var, gamma, dx);

  share_mem_size = thread_per_block / WARP_SIZE * 2 * sizeof(T);
  // GammaAndBetaPropKernel<<<col_dim, thread_per_block, share_mem_size, stream>>>(row_dim, col_dim, epsilon, dy, x,
  // mean,
  //                                                                               var, dg, db);
  int param_reduce_dim = row_dim * col_dim / param_dim;
  GammaAndBetaPropKernel<<<param_dim, thread_per_block, share_mem_size, stream>>>(param_reduce_dim, param_dim, col_dim,
                                                                                  epsilon, dy, x, mean, var, dg, db);
}

template CUDA_LIB_EXPORT void LayerNormGrad(const int &row_dim, const int &col_dim, const int &param_dim,
                                            const float &epsilon, const float *dy, const float *x, const float *mean,
                                            const float *var, const float *gamma, float *dx, float *dg, float *db,
                                            cudaStream_t stream);
template CUDA_LIB_EXPORT void LayerNormGrad(const int &row_dim, const int &col_dim, const int &param_dim,
                                            const half &epsilon, const half *dy, const half *x, const half *mean,
                                            const half *var, const half *gamma, half *dx, half *dg, half *db,
                                            cudaStream_t stream);
