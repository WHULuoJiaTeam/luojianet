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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FTRL_GPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FTRL_GPU_KERNEL_H_

#include <vector>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/ftrl_impl.cuh"
namespace luojianet_ms {
namespace kernel {
constexpr size_t INPUT_NUM = 8;
template <typename T>
class FtrlGpuKernelMod : public NativeGpuKernelMod {
 public:
  FtrlGpuKernelMod()
      : variable_size_(0),
        accumulation_size_(0),
        linear_size_(0),
        gradient_size_(0),
        learning_rate_size_(0),
        l1_regularization_size_(0),
        l2_regularization_size_(0),
        learning_rate_power_size_(0),
        is_null_input_(false),
        kernel_name_("ApplyFtrl") {}

  ~FtrlGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *variable = GetDeviceAddress<T>(inputs, 0);
    T *accumulation = GetDeviceAddress<T>(inputs, 1);
    T *linear = GetDeviceAddress<T>(inputs, 2);
    T *gradient = GetDeviceAddress<T>(inputs, 3);
    T *learning_rate = GetDeviceAddress<T>(inputs, 4);
    T *l1_regularization = GetDeviceAddress<T>(inputs, 5);
    T *l2_regularization = GetDeviceAddress<T>(inputs, 6);
    T *learning_rate_power = GetDeviceAddress<T>(inputs, 7);
    ApplyFtrl(inputs[0]->size / sizeof(T), gradient, learning_rate, l1_regularization, l2_regularization,
              learning_rate_power, variable, accumulation, linear, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    kernel_node_ = kernel_node;
    if (input_num != INPUT_NUM) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be " << INPUT_NUM << ", but got "
                        << input_num;
    }

    variable_size_ = sizeof(T);
    accumulation_size_ = sizeof(T);
    linear_size_ = sizeof(T);
    gradient_size_ = sizeof(T);
    learning_rate_size_ = sizeof(T);
    l1_regularization_size_ = sizeof(T);
    l2_regularization_size_ = sizeof(T);
    learning_rate_power_size_ = sizeof(T);

    auto variable_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto accumulation_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto linear_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    auto gradient_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
    is_null_input_ = CHECK_SHAPE_NULL(variable_shape, kernel_name_, "var") ||
                     CHECK_SHAPE_NULL(accumulation_shape, kernel_name_, "accum") ||
                     CHECK_SHAPE_NULL(linear_shape, kernel_name_, "linear") ||
                     CHECK_SHAPE_NULL(gradient_shape, kernel_name_, "grad");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    for (size_t i = 0; i < variable_shape.size(); i++) {
      variable_size_ *= variable_shape[i];
    }

    for (size_t i = 0; i < accumulation_shape.size(); i++) {
      accumulation_size_ *= accumulation_shape[i];
    }

    for (size_t i = 0; i < linear_shape.size(); i++) {
      linear_size_ *= linear_shape[i];
    }

    for (size_t i = 0; i < gradient_shape.size(); i++) {
      gradient_size_ *= gradient_shape[i];
    }

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(variable_size_);
    input_size_list_.push_back(accumulation_size_);
    input_size_list_.push_back(linear_size_);
    input_size_list_.push_back(gradient_size_);
    input_size_list_.push_back(learning_rate_size_);
    input_size_list_.push_back(l1_regularization_size_);
    input_size_list_.push_back(l2_regularization_size_);
    input_size_list_.push_back(learning_rate_power_size_);
    output_size_list_.push_back(0);
  }

 private:
  size_t variable_size_;
  size_t accumulation_size_;
  size_t linear_size_;
  size_t gradient_size_;
  size_t learning_rate_size_;
  size_t l1_regularization_size_;
  size_t l2_regularization_size_;
  size_t learning_rate_power_size_;
  bool is_null_input_;
  std::string kernel_name_;
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FTRL_GPU_KERNEL_H_
