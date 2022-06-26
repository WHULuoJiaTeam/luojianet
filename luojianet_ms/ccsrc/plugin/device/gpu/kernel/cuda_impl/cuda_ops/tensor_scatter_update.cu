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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/tensor_scatter_update.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T, typename S>
__global__ void TensorScatterUpdateKernel(T *input, S *indices, T *update, T *output, const size_t block_size,
                                          const size_t input_size, const size_t output_size, const size_t indices_dim_0,
                                          const size_t indices_dim_1, S *indices_stride, S *work_shape) {
  int i, j;
  for (size_t read_index = blockIdx.x * blockDim.x + threadIdx.x; read_index < input_size;
       read_index += blockDim.x * gridDim.x) {
    size_t write_index = 0;
    bool out_bound = false;

    i = read_index / block_size;
    j = read_index % block_size;

    for (size_t k = 0; k < indices_dim_1; k++) {
      S indices_i = indices[i * indices_dim_1 + k];
      out_bound |= indices_i >= work_shape[k];
      write_index += indices_i * indices_stride[k];
    }

    write_index += j;
    out_bound |= write_index >= output_size;

    if (!out_bound) {
      output[write_index] = update[read_index];
    }
  }
}

template <typename T, typename S>
void TensorScatterUpdate(T *input, S *indices, T *update, T *output, const size_t &block_size, const size_t &input_size,
                         const size_t &output_size, const size_t &indices_dim_0, const size_t &indices_dim_1,
                         S *indices_stride, S *work_shape, cudaStream_t stream) {
  TensorScatterUpdateKernel<<<GET_BLOCKS(output_size), GET_THREADS, 0, stream>>>(
    input, indices, update, output, block_size, input_size, output_size, indices_dim_0, indices_dim_1, indices_stride,
    work_shape);
  return;
}

template CUDA_LIB_EXPORT void TensorScatterUpdate<half, int>(half *input, int *indices, half *update, half *output,
                                                             const size_t &block_size, const size_t &input_size,
                                                             const size_t &output_size, const size_t &indices_dim_0,
                                                             const size_t &indices_dim_1, int *indices_stride,
                                                             int *work_shape, cudaStream_t stream);
template CUDA_LIB_EXPORT void TensorScatterUpdate<float, int>(float *input, int *indices, float *update, float *output,
                                                              const size_t &block_size, const size_t &input_size,
                                                              const size_t &output_size, const size_t &indices_dim_0,
                                                              const size_t &indices_dim_1, int *indices_stride,
                                                              int *work_shape, cudaStream_t stream);
template CUDA_LIB_EXPORT void TensorScatterUpdate<double, int>(double *input, int *indices, double *update,
                                                               double *output, const size_t &block_size,
                                                               const size_t &input_size, const size_t &output_size,
                                                               const size_t &indices_dim_0, const size_t &indices_dim_1,
                                                               int *indices_stride, int *work_shape,
                                                               cudaStream_t stream);
template CUDA_LIB_EXPORT void TensorScatterUpdate<char, int>(char *input, int *indices, char *update, char *output,
                                                             const size_t &block_size, const size_t &input_size,
                                                             const size_t &output_size, const size_t &indices_dim_0,
                                                             const size_t &indices_dim_1, int *indices_stride,
                                                             int *work_shape, cudaStream_t stream);
template CUDA_LIB_EXPORT void TensorScatterUpdate<unsigned char, int>(unsigned char *input, int *indices,
                                                                      unsigned char *update, unsigned char *output,
                                                                      const size_t &block_size,
                                                                      const size_t &input_size,
                                                                      const size_t &output_size,
                                                                      const size_t &indices_dim_0,
                                                                      const size_t &indices_dim_1, int *indices_stride,
                                                                      int *work_shape, cudaStream_t stream);
template CUDA_LIB_EXPORT void TensorScatterUpdate<int, int>(int *input, int *indices, int *update, int *output,
                                                            const size_t &block_size, const size_t &input_size,
                                                            const size_t &output_size, const size_t &indices_dim_0,
                                                            const size_t &indices_dim_1, int *indices_stride,
                                                            int *work_shape, cudaStream_t stream);
template CUDA_LIB_EXPORT void TensorScatterUpdate<bool, int>(bool *input, int *indices, bool *update, bool *output,
                                                             const size_t &block_size, const size_t &input_size,
                                                             const size_t &output_size, const size_t &indices_dim_0,
                                                             const size_t &indices_dim_1, int *indices_stride,
                                                             int *work_shape, cudaStream_t stream);
template CUDA_LIB_EXPORT void TensorScatterUpdate<bool, int64_t>(bool *input, int64_t *indices, bool *update,
                                                                 bool *output, const size_t &block_size,
                                                                 const size_t &input_size, const size_t &output_size,
                                                                 const size_t &indices_dim_0,
                                                                 const size_t &indices_dim_1, int64_t *indices_stride,
                                                                 int64_t *work_shape, cudaStream_t stream);
template CUDA_LIB_EXPORT void TensorScatterUpdate<float, int64_t>(float *input, int64_t *indices, float *update,
                                                                  float *output, const size_t &block_size,
                                                                  const size_t &input_size, const size_t &output_size,
                                                                  const size_t &indices_dim_0,
                                                                  const size_t &indices_dim_1, int64_t *indices_stride,
                                                                  int64_t *work_shape, cudaStream_t stream);
template CUDA_LIB_EXPORT void TensorScatterUpdate<double, int64_t>(double *input, int64_t *indices, double *update,
                                                                   double *output, const size_t &block_size,
                                                                   const size_t &input_size, const size_t &output_size,
                                                                   const size_t &indices_dim_0,
                                                                   const size_t &indices_dim_1, int64_t *indices_stride,
                                                                   int64_t *work_shape, cudaStream_t stream);
