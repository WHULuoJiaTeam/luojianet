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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BN_TRAINING_UPDATE_GRAD_GPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BN_TRAINING_UPDATE_GRAD_GPU_KERNEL_H_

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/bn_training_update_grad_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace luojianet_ms {
namespace kernel {
constexpr size_t kNIndex = 0;
constexpr size_t kCIndex = 1;
constexpr size_t kHIndex = 2;
constexpr size_t kWIndex = 3;

#if 0
template <typename T>
class BNTraingUpdateGradGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  BNTraingUpdateGradGpuKernelMod() { ResetResource(); }
  ~BNTraingUpdateGradGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }

    T *grads = GetDeviceAddress<T>(inputs, 0);
    T *x = GetDeviceAddress<T>(inputs, 1);
    float *batch_mean = GetDeviceAddress<float>(inputs, 2);
    float *batch_variance = GetDeviceAddress<float>(inputs, 3);

    float *diff_scale = GetDeviceAddress<float>(outputs, 0);
    float *diff_offset = GetDeviceAddress<float>(outputs, 1);

    BNTrainingUpdateGrad(N_, C_, H_, W_, grads, x, batch_mean, batch_variance, diff_scale, diff_offset, epsilon_,
                         reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;

    epsilon_ = GetAttr<float>(kernel_node, "epsilon");

    auto shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    N_ = SizeToInt(shape[kNIndex]);
    C_ = SizeToInt(shape[kCIndex]);
    H_ = SizeToInt(shape[kHIndex]);
    W_ = SizeToInt(shape[kWIndex]);

    auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "x");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    x_size_ = 1;
    for (size_t i = 0; i < input_shape.size(); i++) {
      x_size_ *= input_shape[i];
    }
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    N_ = 0;
    C_ = 0;
    H_ = 0;
    W_ = 0;
    x_size_ = 0;
    epsilon_ = 1e-5;
    input_size_list_.clear();
    output_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(x_size_ * sizeof(T));
    input_size_list_.push_back(x_size_ * sizeof(T));
    input_size_list_.push_back(C_ * sizeof(float));
    input_size_list_.push_back(C_ * sizeof(float));

    output_size_list_.push_back(C_ * sizeof(float));
    output_size_list_.push_back(C_ * sizeof(float));
  }

 private:
  int N_;
  int C_;
  int H_;
  int W_;
  float epsilon_;

  size_t x_size_;
  bool is_null_input_;
};

#endif

}  // namespace kernel
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BN_TRAINING_UPDATE_GRAD_GPU_KERNEL_H_
