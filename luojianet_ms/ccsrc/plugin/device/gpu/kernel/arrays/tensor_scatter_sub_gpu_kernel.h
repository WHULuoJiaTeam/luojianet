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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TENSOR_SCATTER_SUB_GPU_KERNEL_H
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TENSOR_SCATTER_SUB_GPU_KERNEL_H

#include <vector>
#include <algorithm>

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/tensor_scatter_sub.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace luojianet_ms {
namespace kernel {
template <typename T, typename S>
class TensorScatterSubGpuKernelMod : public NativeGpuKernelMod {
 public:
  TensorScatterSubGpuKernelMod()
      : input_size_(1),
        update_size_(1),
        indices_size_(1),
        output_size_(1),
        block_size_(1),
        indices_stride_(nullptr),
        work_shape_(nullptr),
        indices_dim_0_(0),
        indices_dim_1_(0),
        memcpy_flag_(false) {}
  ~TensorScatterSubGpuKernelMod() {
    if (indices_stride_ != nullptr) {
      device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(static_cast<void *>(indices_stride_));
    }
    if (work_shape_ != nullptr) {
      device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(static_cast<void *>(work_shape_));
    }
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    VARIABLE_NOT_USED(workspace);
    T *input = GetDeviceAddress<T>(inputs, 0);
    S *indices = GetDeviceAddress<S>(inputs, 1);
    T *update = GetDeviceAddress<T>(inputs, 2);
    T *output = GetDeviceAddress<T>(outputs, 0);

    if (!memcpy_flag_) {
      const size_t indices_len = sizeof(S) * vec_indices_stride_.size();
      const size_t vec_work_len = sizeof(S) * vec_work_shape_.size();
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(indices_stride_, &vec_indices_stride_[0], indices_len,
                                                 cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpy failed in TensorScatterSubGpuKernelMod::Launch.");
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(work_shape_, &vec_work_shape_[0], vec_work_len, cudaMemcpyHostToDevice,
                                                 reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpy failed in TensorScatterSubGpuKernelMod::Launch.");
      memcpy_flag_ = true;
    }

    const size_t update_size = update_size_ / sizeof(T);
    const size_t output_size = output_size_ / sizeof(T);

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(&output[0], &input[0], input_size_, cudaMemcpyDeviceToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync output failed");

    TensorScatterSub(input, indices, update, output, block_size_, update_size, output_size, indices_dim_0_,
                     indices_dim_1_, indices_stride_, work_shape_, reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    memcpy_flag_ = false;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 3) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 3, but got " << input_num;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 1, but got " << output_num;
    }

    update_shapes_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    indices_shapes_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    input_shapes_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    output_shapes_ = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);

    std::vector<size_t> shape_me = input_shapes_;
    (void)std::transform(shape_me.begin(), shape_me.end(), std::back_inserter(vec_work_shape_),
                         [](const size_t &value) { return static_cast<S>(value); });

    GetSize();

    const size_t indices_len = sizeof(S) * vec_indices_stride_.size();
    void *indices_stride_work = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(indices_len);
    if (indices_stride_work == nullptr) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name
                        << "', the memory alloc of indices_stride_work should be successful, but failed, got size: "
                        << indices_len;
    }
    indices_stride_ = static_cast<S *>(indices_stride_work);

    const size_t vec_work_len = sizeof(S) * vec_work_shape_.size();
    void *work_shape_work = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(vec_work_len);
    if (work_shape_work == nullptr) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name
                        << "', the memory alloc of work_shape_work should be successful, but failed, got size: "
                        << vec_work_len;
    }
    work_shape_ = static_cast<S *>(work_shape_work);

    InitSizeLists();

    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(indices_size_);
    input_size_list_.push_back(update_size_);
    output_size_list_.push_back(output_size_);
    return;
  }

  void GetSize() {
    input_size_ = sizeof(T);
    for (size_t i = 0; i < input_shapes_.size(); i++) {
      input_size_ *= input_shapes_[i];
    }

    indices_size_ = sizeof(S);
    for (size_t i = 0; i < indices_shapes_.size(); i++) {
      indices_size_ *= indices_shapes_[i];
    }
    update_size_ = sizeof(T);
    for (size_t i = 0; i < update_shapes_.size(); i++) {
      update_size_ *= update_shapes_[i];
    }
    output_size_ = sizeof(T);
    for (size_t i = 0; i < output_shapes_.size(); i++) {
      output_size_ *= output_shapes_[i];
    }

    // calculate indices dim 0/1
    indices_dim_0_ = indices_shapes_[0];
    indices_dim_1_ = indices_shapes_[indices_shapes_.size() - 1];

    // calculate block_size
    for (size_t i = indices_dim_1_; i < output_shapes_.size(); i++) {
      block_size_ *= output_shapes_[i];
    }

    // calculate indices_stride
    vec_indices_stride_.resize(indices_dim_1_, 0);
    vec_indices_stride_[indices_dim_1_ - 1] = block_size_;

    for (size_t i = indices_dim_1_ - 1; i > 0; --i) {
      vec_indices_stride_[i - 1] = vec_indices_stride_[i] * output_shapes_[i];
    }
  }

 private:
  std::vector<size_t> update_shapes_;
  std::vector<size_t> indices_shapes_;
  std::vector<size_t> input_shapes_;
  std::vector<size_t> output_shapes_;
  std::vector<S> vec_indices_stride_;
  std::vector<S> vec_work_shape_;

  size_t input_size_;
  size_t update_size_;
  size_t indices_size_;
  size_t output_size_;
  size_t block_size_;

  S *indices_stride_;
  S *work_shape_;
  size_t indices_dim_0_;
  size_t indices_dim_1_;
  bool memcpy_flag_;
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TENSOR_SCATTER_SUB_GPU_KERNEL_H
