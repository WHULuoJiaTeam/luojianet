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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BN_TRAINING_REDUCE_GPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BN_TRAINING_REDUCE_GPU_KERNEL_H_

#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/bn_training_reduce_impl.cuh"

namespace luojianet_ms {
namespace kernel {
constexpr size_t kXshapeSize = 4;
constexpr size_t kBatchIndex = 0;
constexpr size_t kChannelIndex = 1;
constexpr size_t kHeightIndex = 2;
constexpr size_t kWidthIndex = 3;

template <typename T>
class BNTrainingReduceGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  BNTrainingReduceGpuKernelMod() { ResetResource(); }
  ~BNTrainingReduceGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *sum_addr = GetDeviceAddress<T>(outputs, 0);
    T *square_sum_addr = GetDeviceAddress<T>(outputs, 1);
    BNTrainingReduce(n_, c_, h_, w_, input_addr, sum_addr, square_sum_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    InitResource();
    input_size_ = sizeof(T);
    auto x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    if (x_shape.size() != kXshapeSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of input should be 4, but got "
                        << x_shape.size();
    }
    is_null_input_ = CHECK_SHAPE_NULL(x_shape, kernel_name, "x");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    n_ = SizeToInt(x_shape[kBatchIndex]);
    c_ = SizeToInt(x_shape[kChannelIndex]);
    h_ = SizeToInt(x_shape[kHeightIndex]);
    w_ = SizeToInt(x_shape[kWidthIndex]);
    for (auto dim : x_shape) {
      input_size_ *= dim;
    }
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = 0;
    is_null_input_ = false;
    output_size_ = 0;
    input_size_list_.clear();
    output_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);

    output_size_ = c_;
    output_size_list_.push_back(output_size_ * sizeof(float));
    output_size_list_.push_back(output_size_ * sizeof(float));
  }

 private:
  int n_;
  int c_;
  int h_;
  int w_;
  bool is_null_input_;

  size_t input_size_;
  size_t output_size_;
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BN_TRAINING_REDUCE_GPU_KERNEL_H_
