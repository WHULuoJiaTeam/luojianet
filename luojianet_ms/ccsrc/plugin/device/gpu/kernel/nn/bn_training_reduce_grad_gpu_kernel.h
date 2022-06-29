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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BN_TRAINING_REDUCE_GRAD_GPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BN_TRAINING_REDUCE_GRAD_GPU_KERNEL_H_

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/bn_training_reduce_grad_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"


namespace luojianet_ms {
namespace kernel {
constexpr size_t kInputSize = 7;
constexpr size_t kBatchIndex = 0;
constexpr size_t kChannelIndex = 1;
constexpr size_t kHeightIndex = 2;
constexpr size_t kWidthIndex = 3;

template <typename T>
class BNTraingReduceGradGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  BNTraingReduceGradGpuKernelMod() { ResetResource(); }
  ~BNTraingReduceGradGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *grads = GetDeviceAddress<T>(inputs, 0);
    T *x = GetDeviceAddress<T>(inputs, 1);
    T *diff_scale = GetDeviceAddress<T>(inputs, 2);
    T *diff_offset = GetDeviceAddress<T>(inputs, 3);
    T *scale = GetDeviceAddress<T>(inputs, 4);
    T *batch_mean = GetDeviceAddress<T>(inputs, 5);
    T *batch_variance = GetDeviceAddress<T>(inputs, 6);
    T *y = GetDeviceAddress<T>(outputs, 0);
    BNTrainingReduceGrad(grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance, y, epsilon_, batch_,
                         channel_, height_, width_, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    (void)CheckIONumber(kernel_node);

    epsilon_ = GetAttr<float>(kernel_node, "epsilon");

    auto shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    batch_ = SizeToInt(shape[kBatchIndex]);
    channel_ = SizeToInt(shape[kChannelIndex]);
    height_ = SizeToInt(shape[kHeightIndex]);
    width_ = SizeToInt(shape[kWidthIndex]);

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
    batch_ = 0;
    channel_ = 0;
    height_ = 0;
    width_ = 0;
    x_size_ = 0;
    para_size_ = 0;
    epsilon_ = 1e-5;
    input_size_list_.clear();
    output_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(x_size_ * sizeof(T));
    input_size_list_.push_back(x_size_ * sizeof(float));
    input_size_list_.push_back(channel_ * sizeof(float));
    input_size_list_.push_back(channel_ * sizeof(float));
    input_size_list_.push_back(channel_ * sizeof(float));
    input_size_list_.push_back(channel_ * sizeof(float));
    input_size_list_.push_back(channel_ * sizeof(float));
    input_size_list_.push_back(sizeof(float));
    output_size_list_.push_back(x_size_ * sizeof(T));
  }

 private:
  void CheckIONumber(const CNodePtr &kernel_node) {
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != kInputSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 7, but got " << input_num;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << output_num;
    }
  }

  int batch_;
  int channel_;
  int height_;
  int width_;
  float epsilon_;

  size_t x_size_;
  size_t para_size_;
  size_t output_size_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace luojianet_ms
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BN_TRAINING_REDUCE_GRAD_GPU_KERNEL_H_
