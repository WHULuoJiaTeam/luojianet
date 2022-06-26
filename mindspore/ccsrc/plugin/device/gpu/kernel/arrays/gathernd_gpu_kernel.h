/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_GATHERND_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_GATHERND_GPU_KERNEL_H_

#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/gathernd.cuh"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class GatherNdFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  GatherNdFwdGpuKernelMod() : dev_batch_strides_(nullptr), dev_batch_indices_(nullptr), memcpy_flag_(false) {}
  ~GatherNdFwdGpuKernelMod() {
    if (dev_batch_strides_ != nullptr) {
      device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(static_cast<void *>(dev_batch_strides_));
    }
    if (dev_batch_indices_ != nullptr) {
      device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(static_cast<void *>(dev_batch_indices_));
    }
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    VARIABLE_NOT_USED(workspace);
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    S *indices_addr = GetDeviceAddress<S>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    if (!memcpy_flag_) {
      const size_t strides_len = sizeof(S) * batch_strides_.size();
      const size_t indices_len = sizeof(S) * batch_indices_.size();
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(dev_batch_strides_, &batch_strides_[0], strides_len,
                                                 cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync failed in GatherNdFwdGpuKernelMod::Launch.");
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(dev_batch_indices_, &batch_indices_[0], indices_len,
                                                 cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync failed in GatherNdFwdGpuKernelMod::Launch.");
      memcpy_flag_ = true;
    }

    GatherNd(input_addr, indices_addr, output_addr, dims_[0], dims_[1], dims_[2], dev_batch_strides_,
             dev_batch_indices_, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    InitResource();
    memcpy_flag_ = false;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 2, but got " << input_num;
    }
    input_shapes_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    indices_shapes_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    output_shapes_ = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shapes_, kernel_name, "input_x") ||
                     CHECK_SHAPE_NULL(indices_shapes_, kernel_name, "indices") ||
                     CHECK_SHAPE_NULL(output_shapes_, kernel_name, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    Reshape();

    size_t dim_indices_last = dims_[dims_.size() - 1];
    batch_strides_.resize(dim_indices_last, 0);
    batch_indices_.resize(dim_indices_last, 0);

    if (dim_indices_last > 0) {
      batch_strides_[dim_indices_last - 1] = input_shapes_[dim_indices_last - 1];
      batch_indices_[dim_indices_last - 1] = dims_[1];
    }
    for (size_t i = dim_indices_last - 1; i > 0; --i) {
      batch_strides_[i - 1] = input_shapes_[i - 1];
      batch_indices_[i - 1] = batch_indices_[i] * input_shapes_[i];
    }

    const size_t strides_len = sizeof(S) * batch_strides_.size();
    void *dev_batch_strides_work = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(strides_len);
    if (dev_batch_strides_work == nullptr) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name
                        << "', the memory alloc of dev_batch_strides_work should be successful, but failed, got size: "
                        << strides_len;
    }
    dev_batch_strides_ = static_cast<S *>(dev_batch_strides_work);

    const size_t indices_len = sizeof(S) * batch_indices_.size();
    void *dev_batch_indices_work = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(indices_len);
    if (dev_batch_indices_work == nullptr) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name
                        << "', the memory alloc of dev_batch_indices_work should be successful, but failed, got size: "
                        << indices_len;
    }
    dev_batch_indices_ = static_cast<S *>(dev_batch_indices_work);

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    size_t size = common::AnfAlgo::TensorSizeInByte<T>(input_shapes_);
    input_size_list_.push_back(size);

    size = common::AnfAlgo::TensorSizeInByte<T>(indices_shapes_);
    input_size_list_.push_back(size);

    size = common::AnfAlgo::TensorSizeInByte<T>(output_shapes_);
    output_size_list_.push_back(size);
  }

 private:
  void Reshape() {
    size_t dim_of_indices = 1;
    for (size_t i = 0; i < indices_shapes_.size() - IntToSize(1); i++) {
      dim_of_indices *= indices_shapes_[i];
    }

    size_t dim_after_indices = 1;
    size_t dim_indices_last = indices_shapes_[indices_shapes_.size() - IntToSize(1)];
    for (size_t i = dim_indices_last; i < input_shapes_.size(); i++) {
      dim_after_indices *= input_shapes_[i];
    }
    dims_.emplace_back(dim_of_indices);
    dims_.emplace_back(dim_after_indices);
    dims_.emplace_back(dim_indices_last);
    return;
  }

  std::vector<size_t> input_shapes_;
  std::vector<size_t> indices_shapes_;
  std::vector<size_t> output_shapes_;

  std::vector<size_t> dims_;

  std::vector<S> batch_strides_;
  std::vector<S> batch_indices_;

  S *dev_batch_strides_;
  S *dev_batch_indices_;
  bool memcpy_flag_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_GATHERND_GPU_KERNEL_H_
