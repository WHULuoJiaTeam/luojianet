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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FUSED_SCALE_MOMENTUM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FUSED_SCALE_MOMENTUM_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/momentum_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T, typename S>
class CombineMomentumGpuKernelMod : public NativeGpuKernelMod {
 public:
  CombineMomentumGpuKernelMod()
      : element_num_(1), num_(0), input_num_(6), is_null_input_(false), kernel_name_("CombineMomentum") {}
  ~CombineMomentumGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    for (size_t i = 0; i < num_; i++) {
      if (input_num_ == 6) {
        T *scale = GetDeviceAddress<T>(inputs, i * input_num_);
        T *variable = GetDeviceAddress<T>(inputs, i * input_num_ + 1);
        T *acc = GetDeviceAddress<T>(inputs, i * input_num_ + 2);
        T *lr = GetDeviceAddress<T>(inputs, i * input_num_ + 3);
        S *grad = GetDeviceAddress<S>(inputs, i * input_num_ + 4);
        T *mom = GetDeviceAddress<T>(inputs, i * input_num_ + 5);
        FusedScaleMomentum(elements_[i], scale, variable, acc, lr, grad, mom, stream);
      } else {
        T *weight_decay = GetDeviceAddress<T>(inputs, i * input_num_);
        T *scale = GetDeviceAddress<T>(inputs, i * input_num_ + 1);
        T *variable = GetDeviceAddress<T>(inputs, i * input_num_ + 2);
        T *acc = GetDeviceAddress<T>(inputs, i * input_num_ + 3);
        T *lr = GetDeviceAddress<T>(inputs, i * input_num_ + 4);
        S *grad = GetDeviceAddress<S>(inputs, i * input_num_ + 5);
        T *mom = GetDeviceAddress<T>(inputs, i * input_num_ + 6);
        FusedWeightDecayScaleMomentum(elements_[i], weight_decay, scale, variable, acc, lr, grad, mom, stream);
      }
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    num_ = GetAttr<size_t>(kernel_node, "n");
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    if (kernel_name == "CombineMomentum") {
      input_num_ = 6;
    } else {
      input_num_ = 7;
    }
    for (size_t i = 0; i < num_; i++) {
      element_num_ = 1;
      auto variable_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, i * input_num_ + input_num_ - 5);
      is_null_input_ = CHECK_SHAPE_NULL(variable_shape, kernel_name_,
                                        "input[" + std::to_string(i * input_num_ + input_num_ - 5) + "]");
      if (is_null_input_) {
        InitSizeLists();
        return true;
      }
      for (size_t j = 0; j < variable_shape.size(); j++) {
        element_num_ *= variable_shape[j];
      }
      elements_.push_back(element_num_);
      InitSizeLists();
    }
    return true;
  }

 protected:
  void InitSizeLists() override {
    if (input_num_ == 7) {
      input_size_list_.push_back(sizeof(T));
    }
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
  std::vector<size_t> elements_;
  size_t num_;
  int input_num_;
  bool is_null_input_;
  std::string kernel_name_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FUSED_SCALE_MOMENTUM_GPU_KERNEL_H_
