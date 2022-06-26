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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DYNAMIC_SHAPE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DYNAMIC_SHAPE_GPU_KERNEL_H_

#include <cuda_runtime.h>

#include <vector>

#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class TensorShapeGpuKernelMod : public NativeGpuKernelMod {
 public:
  TensorShapeGpuKernelMod() { ResetResource(); }
  ~TensorShapeGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    S *output_device_address = GetDeviceAddress<S>(outputs, 0);
    size_t prev_node_output_shape_size = prev_node_output_shape_.size() * sizeof(S);
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_,
      cudaMemcpyAsync(output_device_address, prev_node_output_shape_.data(), prev_node_output_shape_size,
                      cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync prev_node_output_shape failed");

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    size_t input_count = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_count != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 1, but got " << input_count;
    }

    std::vector<size_t> prev_node_output_shape_tmp = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(prev_node_output_shape_tmp, kernel_name, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    input_size_ = 1;
    for (const size_t &e : prev_node_output_shape_tmp) {
      input_size_ *= e;
      // shapes are Tensors with elements of type S (int32, or int64) but
      // GetPrevNodeOutputInferShape returns vector of size_t, so we use
      // an S* for allocated output memory and cast to an integral type here,
      // otherwise the memcpy will fail silently.
      prev_node_output_shape_.push_back(e);
    }

    output_size_ = prev_node_output_shape_.size();

    InitSizeLists();

    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = 0;
    output_size_ = 0;
    is_null_input_ = false;
    prev_node_output_shape_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    output_size_list_.push_back(output_size_ * sizeof(S));
  }

 private:
  size_t input_size_;
  size_t output_size_;
  bool is_null_input_;
  std::vector<S> prev_node_output_shape_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DYNAMIC_SHAPE_GPU_KERNEL_H_
