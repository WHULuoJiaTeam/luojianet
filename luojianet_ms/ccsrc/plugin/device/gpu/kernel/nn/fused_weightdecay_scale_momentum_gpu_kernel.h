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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FUSED_WEIGHTDECAY_SCALE_MOMENTUM_GPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FUSED_WEIGHTDECAY_SCALE_MOMENTUM_GPU_KERNEL_H_

#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/momentum_impl.cuh"
namespace luojianet_ms {
namespace kernel {
constexpr size_t INPUT_NUM = 7;
template <typename T, typename S>
class FusedWeightDecayScaleMomentumGpuKernelMod : public NativeGpuKernelMod {
 public:
  FusedWeightDecayScaleMomentumGpuKernelMod() : element_num_(1), is_null_input_(false) {}
  ~FusedWeightDecayScaleMomentumGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *weight_decay = GetDeviceAddress<T>(inputs, 0);
    T *scale = GetDeviceAddress<T>(inputs, 1);
    T *variable = GetDeviceAddress<T>(inputs, 2);
    T *accumulation = GetDeviceAddress<T>(inputs, 3);
    T *learning_rate = GetDeviceAddress<T>(inputs, 4);
    S *gradient = GetDeviceAddress<S>(inputs, 5);
    T *momentum = GetDeviceAddress<T>(inputs, 6);

    FusedWeightDecayScaleMomentum(element_num_, weight_decay, scale, variable, accumulation, learning_rate, gradient,
                                  momentum, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    kernel_node_ = kernel_node;
    if (input_num != INPUT_NUM) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be " << INPUT_NUM << ", but got "
                        << input_num;
    }

    auto variable_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    is_null_input_ = CHECK_SHAPE_NULL(variable_shape, kernel_name, "variable");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    for (size_t i = 0; i < variable_shape.size(); i++) {
      element_num_ *= variable_shape[i];
    }

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(sizeof(T));
    input_size_list_.push_back(sizeof(T));
    input_size_list_.push_back(element_num_ * sizeof(T));
    input_size_list_.push_back(element_num_ * sizeof(T));
    input_size_list_.push_back(sizeof(T));
    input_size_list_.push_back(element_num_ * sizeof(S));
    input_size_list_.push_back(sizeof(T));
    output_size_list_.push_back(element_num_ * sizeof(T));
  }

 private:
  size_t element_num_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FUSED_WEIGHTDECAY_SCALE_MOMENTUM_GPU_KERNEL_H_
