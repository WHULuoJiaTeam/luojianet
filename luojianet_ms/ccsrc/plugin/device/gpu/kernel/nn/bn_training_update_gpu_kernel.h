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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BN_TRAINING_UPDATE_GPU_KERNEL_H_
#define LUOJINAET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BN_TRAINING_UPDATE_GPU_KERNEL_H_

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/bn_training_update_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace luojianet_ms {
namespace kernel {
constexpr size_t kNIndex = 0;
constexpr size_t kCIndex = 1;
constexpr size_t kHIndex = 2;
constexpr size_t kWIndex = 3;
constexpr size_t kFactor = 0.1;

template <typename T>
class BNTrainingUpdateGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  BNTrainingUpdateGpuKernelMod() { ResetResource(); }
  ~BNTrainingUpdateGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *x = GetDeviceAddress<T>(inputs, 0);
    float *sum = GetDeviceAddress<float>(inputs, 1);
    float *square_sum = GetDeviceAddress<float>(inputs, 2);
    float *scale = GetDeviceAddress<float>(inputs, 3);
    float *offset = GetDeviceAddress<float>(inputs, 4);
    float *mean = GetDeviceAddress<float>(inputs, 5);
    float *variance = GetDeviceAddress<float>(inputs, 6);

    T *y = GetDeviceAddress<T>(outputs, 0);
    float *mean_output = GetDeviceAddress<float>(outputs, 1);
    float *variance_output = GetDeviceAddress<float>(outputs, 2);
    float *save_mean_reduce_output = GetDeviceAddress<float>(outputs, 3);
    float *save_variance_reduce_output = GetDeviceAddress<float>(outputs, 4);

    BNTrainingUpdate(N_, C_, H_, W_, x, y, sum, square_sum, scale, offset, mean, variance, factor_, epsilon_,
                     mean_output, variance_output, save_mean_reduce_output, save_variance_reduce_output,
                     reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;

    epsilon_ = GetAttr<float>(kernel_node, "epsilon");
    factor_ = GetAttr<float>(kernel_node, "factor");

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
    factor_ = kFactor;
    epsilon_ = 1e-5;
    input_size_list_.clear();
    output_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(x_size_ * sizeof(T));
    input_size_list_.push_back(C_ * sizeof(float));
    input_size_list_.push_back(C_ * sizeof(float));
    input_size_list_.push_back(C_ * sizeof(float));
    input_size_list_.push_back(C_ * sizeof(float));
    input_size_list_.push_back(C_ * sizeof(float));
    input_size_list_.push_back(C_ * sizeof(float));
    output_size_list_.push_back(x_size_ * sizeof(T));
    output_size_list_.push_back(C_ * sizeof(float));
    output_size_list_.push_back(C_ * sizeof(float));
    output_size_list_.push_back(C_ * sizeof(float));
    output_size_list_.push_back(C_ * sizeof(float));
  }

 private:
  float epsilon_;
  float factor_;
  size_t x_size_;
  size_t N_;
  size_t C_;
  size_t H_;
  size_t W_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace luojianet_ms
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BN_TRAINING_UPDATE_GPU_KERNEL_H_
