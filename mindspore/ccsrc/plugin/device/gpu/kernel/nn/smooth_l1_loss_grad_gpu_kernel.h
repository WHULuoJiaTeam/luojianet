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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SMOOTH_L1_LOSS_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SMOOTH_L1_LOSS_GRAD_GPU_KERNEL_H_

#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/smooth_l1_loss_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T>
class SmoothL1LossGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  SmoothL1LossGradGpuKernelMod() : input_size_(1), beta_(1.0) {}
  ~SmoothL1LossGradGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *prediction = GetDeviceAddress<T>(inputs, 0);
    T *target = GetDeviceAddress<T>(inputs, 1);
    T *dloss = GetDeviceAddress<T>(inputs, 2);
    T *dx = GetDeviceAddress<T>(outputs, 0);

    SmoothL1LossGrad(input_size_, beta_, prediction, target, dloss, dx, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    kernel_node_ = kernel_node;
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name, "logits");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= input_shape[i];
    }

    beta_ = GetAttr<float>(kernel_node, "beta");
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    input_size_list_.push_back(input_size_ * sizeof(T));
    output_size_list_.push_back(input_size_ * sizeof(T));
  }

 private:
  size_t input_size_;
  float beta_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SMOOTH_L1_LOSS_GRAD_GPU_KERNEL_H_
