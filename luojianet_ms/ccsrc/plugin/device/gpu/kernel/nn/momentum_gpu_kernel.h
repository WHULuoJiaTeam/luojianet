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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MOMENTUM_GPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MOMENTUM_GPU_KERNEL_H_

#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/momentum_impl.cuh"
namespace luojianet_ms {
namespace kernel {
constexpr size_t INPUT_NUM = 5;
template <typename T, typename S, typename G>
class MomentumGpuKernelMod : public NativeGpuKernelMod {
 public:
  MomentumGpuKernelMod()
      : use_nesterov_(false),
        is_null_input_(false),
        variable_size_(0),
        accumulation_size_(0),
        learning_rate_size_(0),
        gradient_size_(0),
        momentum_size_(0) {}
  ~MomentumGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *variable = GetDeviceAddress<T>(inputs, 0);
    T *accumulation = GetDeviceAddress<T>(inputs, 1);
    S *learning_rate = GetDeviceAddress<S>(inputs, 2);
    G *gradient = GetDeviceAddress<G>(inputs, 3);
    S *momentum = GetDeviceAddress<S>(inputs, 4);
    MomentumUpdateVariable(inputs[0]->size / sizeof(T), variable, accumulation, learning_rate, gradient, momentum,
                           use_nesterov_, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    kernel_node_ = kernel_node;
    if (input_num != INPUT_NUM) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be " << INPUT_NUM << ", but got "
                        << input_num;
    }
    use_nesterov_ = GetAttr<bool>(kernel_node, "use_nesterov");

    variable_size_ = sizeof(T);
    accumulation_size_ = sizeof(T);
    learning_rate_size_ = sizeof(S);
    gradient_size_ = sizeof(G);
    momentum_size_ = sizeof(S);

    auto variable_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto accumulation_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto gradient_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
    is_null_input_ = CHECK_SHAPE_NULL(variable_shape, kernel_name, "variable") ||
                     CHECK_SHAPE_NULL(accumulation_shape, kernel_name, "accumulation") ||
                     CHECK_SHAPE_NULL(gradient_shape, kernel_name, "gradient");
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
    input_size_list_.push_back(learning_rate_size_);
    input_size_list_.push_back(gradient_size_);
    input_size_list_.push_back(momentum_size_);
    output_size_list_.push_back(variable_size_);
  }

 private:
  bool use_nesterov_;
  bool is_null_input_;
  size_t variable_size_;
  size_t accumulation_size_;
  size_t learning_rate_size_;
  size_t gradient_size_;
  size_t momentum_size_;
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MOMENTUM_GPU_KERNEL_H_
