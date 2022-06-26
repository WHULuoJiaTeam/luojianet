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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_OTHER_GPU_CONVERT_TO_DYNAMIC_SHAPE_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_OTHER_GPU_CONVERT_TO_DYNAMIC_SHAPE_GPU_KERNEL_H

#include <vector>

#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class GpuConvertToDynamicShapeGpuKernelMod : public NativeGpuKernelMod {
 public:
  GpuConvertToDynamicShapeGpuKernelMod() { ResetResource(); }
  ~GpuConvertToDynamicShapeGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    VARIABLE_NOT_USED(workspace);
    T *input_device_address = GetDeviceAddress<T>(inputs, 0);
    T *output_device_address = GetDeviceAddress<T>(outputs, 0);
    cuda_stream_ptr_ = stream_ptr;

    CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                              cudaMemcpyAsync(output_device_address, input_device_address, input_size_ * sizeof(T),
                                              cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                              "Failed to copy gpu memory.");

    return true;
  }

  void UpdateOp() override {
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream_ptr_)),
                               "cudaStreamSynchronized failed");

    std::vector<TypeId> output_types = {common::AnfAlgo::GetOutputInferDataType(kernel_node_.lock(), 0)};
    std::vector<std::vector<size_t>> output_shapes = {input_shape_};
    common::AnfAlgo::SetOutputInferTypeAndShape(output_types, output_shapes, kernel_node_.lock().get());
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    MS_EXCEPTION_IF_NULL(kernel_node);
    kernel_node_ = kernel_node;
    size_t input_count = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_count != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 1, but got " << input_count;
    }

    input_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape_, kernel_name, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    for (const size_t &e : input_shape_) {
      input_size_ *= e;
    }

    InitSizeLists();
    is_need_updateop_ = true;
    return true;
  }

  void ResetResource() noexcept override {
    cuda_stream_ptr_ = nullptr;
    input_shape_.clear();
    input_size_ = 1;
    is_null_input_ = false;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    output_size_list_.push_back(input_size_ * sizeof(T));
  }

 private:
  void *cuda_stream_ptr_;
  std::vector<size_t> input_shape_;
  size_t input_size_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_OTHER_GPU_CONVERT_TO_DYNAMIC_SHAPE_GPU_KERNEL_H
