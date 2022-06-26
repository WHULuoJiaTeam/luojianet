/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_EMBEDDING_LOOKUP_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_EMBEDDING_LOOKUP_GPU_KERNEL_H_

#include <vector>
#include <algorithm>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/embedding_lookup_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class EmbeddingLookupKernelMod : public NativeGpuKernelMod {
 public:
  EmbeddingLookupKernelMod() { ResetResource(); }
  ~EmbeddingLookupKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    VARIABLE_NOT_USED(workspace);
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    S *indices_addr = GetDeviceAddress<S>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    if (is_dynamic_shape_) {
      int64_t *offset_device_address = GetDeviceAddress<int64_t>(inputs, 2);  // only get this if in dynamic mode
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(&offset_, offset_device_address, sizeof(int64_t),
                                                 cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync offset_ failed");
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaDeviceSynchronize(),
                                 "cudaDeviceSyncFailed - EmbeddingLookup - in dynamic mode");
    }
    auto input_dim1 = input_shapes_[0];
    CalEmbeddingLookup(input_addr, indices_addr, output_addr, dims_[0], dims_[1], dims_[2], input_dim1, offset_,
                       reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num == 3) {
      is_dynamic_shape_ = true;
      MS_LOG(INFO) << " EmbeddingLookup running in Dynamic Mode.";
    } else if (input_num == 2) {
      MS_LOG(INFO) << " EmbeddingLookup running in Normal Mode.";
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 2 or 3, but got " << input_num;
    }
    input_shapes_ = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 0);
    indices_shapes_ = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 1);
    output_shapes_ = AnfAlgo::GetOutputDeviceShapeAdaptively(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shapes_, kernel_name, "input") ||
                     CHECK_SHAPE_NULL(indices_shapes_, kernel_name, "input_indices") ||
                     CHECK_SHAPE_NULL(output_shapes_, kernel_name, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    if (input_shapes_.size() < 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of input cannot be less than 1, but got "
                        << input_shapes_.size();
    }
    if (!is_dynamic_shape_) {
      offset_ = GetAttr<int64_t>(kernel_node, "offset");
    }
    Reshape();
    InitSizeLists();
    return true;
  }
  void ResetResource() noexcept override {
    is_dynamic_shape_ = false;
    is_null_input_ = false;
    input_shapes_.clear();
    indices_shapes_.clear();
    output_shapes_.clear();
    std::fill(dims_, dims_ + 3, 0);
    offset_ = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    size_t size = GetSize(input_shapes_);
    input_size_list_.push_back(size);
    size = GetSize(indices_shapes_);
    input_size_list_.push_back(size);
    if (is_dynamic_shape_) {
      input_size_list_.push_back(sizeof(int64_t));
    }
    size = GetSize(output_shapes_);
    output_size_list_.push_back(size);
  }

 private:
  void Reshape() {
    int64_t axis = 0;
    size_t dim_before_axis = 1;
    for (size_t i = 0; i < LongToSize(axis); i++) {
      dim_before_axis *= output_shapes_[i];
    }
    size_t dim_of_indices = 1;
    for (size_t i = 0; i < indices_shapes_.size(); i++) {
      dim_of_indices *= indices_shapes_[i];
    }
    size_t dim_after_indices = 1;
    for (size_t i = LongToSize(axis) + indices_shapes_.size(); i < output_shapes_.size(); i++) {
      dim_after_indices *= output_shapes_[i];
    }
    dims_[0] = dim_before_axis;
    dims_[1] = dim_of_indices;
    dims_[2] = dim_after_indices;
    return;
  }
  size_t GetSize(const std::vector<size_t> &shape) const {
    if (shape.size() == 0) {
      return 0;
    }
    size_t result = sizeof(T);
    for (size_t i = 0; i < shape.size(); i++) {
      result *= shape[i];
    }
    return result;
  }

  std::vector<size_t> input_shapes_;
  std::vector<size_t> indices_shapes_;
  std::vector<size_t> output_shapes_;
  size_t dims_[3] = {};
  int64_t offset_;
  bool is_dynamic_shape_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_EMBEDDING_LOOKUP_GPU_KERNEL_H_
