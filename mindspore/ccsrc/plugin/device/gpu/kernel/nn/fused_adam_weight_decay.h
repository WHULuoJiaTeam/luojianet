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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FUSED_ADAM_WEIGHT_DECAY_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FUSED_ADAM_WEIGHT_DECAY_KERNEL_H_

#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adam_weight_decay_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class FusedAdamWeightDecayGpuKernelMod : public NativeGpuKernelMod {
 public:
  FusedAdamWeightDecayGpuKernelMod() : element_nums_(0), weight_decay_(false), is_null_input_(false) {}
  ~FusedAdamWeightDecayGpuKernelMod() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    auto node_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    if (node_name == "FusedAdamWeightDecay") {
      weight_decay_ = true;
    }

    auto shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 7);
    is_null_input_ = CHECK_SHAPE_NULL(shape, node_name, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    element_nums_ = 1;
    for (auto i : shape) {
      element_nums_ *= i;
    }

    InitSizeLists();
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    float *beta1 = GetDeviceAddress<float>(inputs, 0);
    float *one_sub_beta1 = GetDeviceAddress<float>(inputs, 1);
    float *beta2 = GetDeviceAddress<float>(inputs, 2);
    float *one_sub_beta2 = GetDeviceAddress<float>(inputs, 3);
    float *epsilon = GetDeviceAddress<float>(inputs, 4);
    float *lr = GetDeviceAddress<float>(inputs, 5);
    T *param = GetDeviceAddress<T>(inputs, 6);
    T *m = GetDeviceAddress<T>(inputs, 7);
    T *v = GetDeviceAddress<T>(inputs, 8);
    T *gradient = GetDeviceAddress<T>(inputs, 9);
    float *weight_decay = weight_decay_ ? GetDeviceAddress<float>(inputs, 10) : nullptr;
    AdamWeightDecay(element_nums_, true, beta1, one_sub_beta1, beta2, one_sub_beta2, epsilon, lr, weight_decay, m, v,
                    param, gradient, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitResource() override{};
  void InitSizeLists() override {
    input_size_list_.push_back(sizeof(float));
    input_size_list_.push_back(sizeof(float));
    input_size_list_.push_back(sizeof(float));
    input_size_list_.push_back(sizeof(float));
    input_size_list_.push_back(element_nums_ * sizeof(T));
    input_size_list_.push_back(sizeof(float));
    input_size_list_.push_back(sizeof(float));
    input_size_list_.push_back(element_nums_ * sizeof(T));
    if (weight_decay_) {
      input_size_list_.push_back(sizeof(float));
    }
    output_size_list_.push_back(element_nums_ * sizeof(T));
  }

 private:
  int element_nums_;
  bool weight_decay_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FUSED_ADAM_WEIGHT_DECAY_KERNEL_H_
