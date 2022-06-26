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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_LINSPACE_GPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_LINSPACE_GPU_KERNEL_H_

#include <vector>
#include <memory>
#include <iostream>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/linspace.cuh"
#include "plugin/device/gpu/kernel/kernel_constants.h"

namespace luojianet_ms {
namespace kernel {
template <typename T>
class LinSpaceGpuKernelMod : public NativeGpuKernelMod {
 public:
  LinSpaceGpuKernelMod() { ResetResource(); }
  ~LinSpaceGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    VARIABLE_NOT_USED(workspace);
    T *start_addr = GetDeviceAddress<T>(inputs, 0);
    T *stop_addr = GetDeviceAddress<T>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    calLinSpace(start_addr, stop_addr, value_count_, output_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    kernel_node_ = kernel_node;
    if (input_num != 3) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 3, but got " << input_num;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 1, but got " << output_num;
    }
    auto input_1 = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 0);
    auto input_2 = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 1);
    auto value_count = AnfAlgo::GetOutputDeviceShapeAdaptively(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_1, kernel_name, "start") ||
                     CHECK_SHAPE_NULL(input_2, kernel_name, "stop") ||
                     CHECK_SHAPE_NULL(value_count, kernel_name, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    // error checking input data
    if ((input_1.size() != 0) || (input_2.size() != 0)) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', both start and end should be 0-D Tensors, but got dimension "
                        << "of start: " << input_1.size() << " and dimension of end: " << input_2.size();
    }

    if (value_count.size() != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of output should be 1, but got "
                        << value_count.size();
    }
    value_count_ = value_count[0];
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    value_count_ = 0;
    is_null_input_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(sizeof(T));  // Scalar tensor
    input_size_list_.push_back(sizeof(T));  // Scalar tensor
    output_size_list_.push_back(value_count_ * sizeof(T));
  }

 private:
  size_t value_count_ = 0;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_LINSPACE_GPU_KERNEL_H_
