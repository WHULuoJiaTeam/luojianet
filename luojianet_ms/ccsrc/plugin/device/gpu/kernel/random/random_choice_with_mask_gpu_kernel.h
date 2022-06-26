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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDOM_CHOICE_WITH_MASK_GPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDOM_CHOICE_WITH_MASK_GPU_KERNEL_H_

#include <vector>
#include <chrono>
#include <random>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/random_choice_with_mask_impl.cuh"

namespace luojianet_ms {
namespace kernel {
template <typename T, typename S>
class RandomChoiceWithMaskGpuKernelMod : public NativeGpuKernelMod {
 public:
  RandomChoiceWithMaskGpuKernelMod()
      : input_shape_size_(0), seed_(0), seed2_(0), input_size_(1), count_(0), ceil_power2_(0), is_null_input_(false) {}
  ~RandomChoiceWithMaskGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input = GetDeviceAddress<T>(inputs, 0);
    S *output_index = GetDeviceAddress<S>(outputs, 0);
    T *output_mask = GetDeviceAddress<T>(outputs, 1);
    int seedc = 0;
    if (seed2_ != 0) {
      seedc = seed2_;
    } else if (seed_ != 0) {
      seedc = seed_;
    } else {
      seedc = generator_();
    }
    if (count_ > kSmallK || input_shape_size_ > 1) {
      S *index_buff = GetDeviceAddress<S>(workspaces, 0);
      S *mask_buff = GetDeviceAddress<S>(workspaces, 1);
      S *rank_buff = GetDeviceAddress<S>(workspaces, 2);
      S *Tnum_buff = GetDeviceAddress<S>(workspaces, 3);
      S *tmp_buff = GetDeviceAddress<S>(workspaces, 4);
      void *States = GetDeviceAddress<void *>(workspaces, 5);
      curandState *devStates = reinterpret_cast<curandState *>(States);
      CalRandomChoiceWithMask(input_size_, input_shape_size_, input_shape_5D_[0], input_shape_5D_[1],
                              input_shape_5D_[2], input_shape_5D_[3], input_shape_5D_[4], seedc, count_, input,
                              output_index, output_mask, index_buff, mask_buff, rank_buff, Tnum_buff, tmp_buff,
                              devStates, reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      CalRandomChoiceWithMaskSmall<float, S, T>(input_size_, seedc, count_, input, output_index, output_mask,
                                                reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    MS_EXCEPTION_IF_NULL(kernel_node);
    uint32_t time_interval = std::chrono::system_clock::now().time_since_epoch().count();
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 1, but got " << input_num;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 2, but got " << output_num;
    }
    auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    input_shape_size_ = input_shape.size();
    if (input_shape_size_ < 1 || input_shape_size_ > MAX_DIMENSION) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of input should be in (1, 5), but got "
                        << input_shape_size_;
    }
    // convert size_t to int
    for (auto i = 0; i < input_shape_size_; i++) {
      input_shape_5D_.push_back(input_shape[i]);
    }
    // convert shape to 5D
    while (input_shape_5D_.size() != MAX_DIMENSION) {
      (void)input_shape_5D_.insert(input_shape_5D_.begin(), 1);
    }
    // init seedc
    seed_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "seed"));
    seed2_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "seed2"));
    generator_.seed(time_interval);
    // init memory
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= input_shape[i];
    }
    count_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "count"));
    // upper ceiling for input for ceil_power2
    if (count_ > kSmallK || input_shape_size_ > 1) {
      ceil_power2_ = RcwmRoundUpPower2(input_size_);
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    output_size_list_.push_back(count_ * input_shape_size_ * sizeof(S));
    output_size_list_.push_back(count_ * sizeof(T));
    if (count_ > kSmallK || input_shape_size_ > 1) {
      workspace_size_list_.push_back(input_size_ * input_shape_size_ * sizeof(S));
      workspace_size_list_.push_back(ceil_power2_ * sizeof(S));
      workspace_size_list_.push_back(ceil_power2_ * sizeof(S));
      int blocknum = std::ceil(static_cast<float>(ceil_power2_) / BLOCKSIZE);
      workspace_size_list_.push_back(blocknum * sizeof(S));
      workspace_size_list_.push_back(ceil_power2_ * sizeof(S));
      workspace_size_list_.push_back(ceil_power2_ * sizeof(curandState));
    }
  }

 private:
  const int kSmallK = 2048;
  int input_shape_size_;
  int seed_;
  int seed2_;
  int input_size_;
  int count_;
  int ceil_power2_;
  bool is_null_input_;
  std::mt19937 generator_;
  std::vector<int> input_shape_5D_;
};
}  // namespace kernel
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDOM_CHOICE_WITH_MASK_GPU_KERNEL_H_
