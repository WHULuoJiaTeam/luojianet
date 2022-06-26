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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_DROPOUT_GPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_DROPOUT_GPU_KERNEL_H_

#include <vector>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/dropout_impl.cuh"
#include "include/curand.h"

namespace luojianet_ms {
namespace kernel {
template <typename T>
class DropoutFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  DropoutFwdGpuKernelMod() { ResetResource(); }
  ~DropoutFwdGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }

    T *input = GetDeviceAddress<T>(inputs, 0);
    T *output = GetDeviceAddress<T>(outputs, 0);
    T *mask = GetDeviceAddress<T>(outputs, 1);
    float *mask_f = GetDeviceAddress<float>(workspace, 0);

    CHECK_CURAND_RET_WITH_EXCEPT(curandSetStream(mask_generator_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "Failed to set stream for generator");
    // curandGen only support float or double for mask.
    CHECK_CURAND_RET_WITH_EXCEPT(curandGenerateUniform(mask_generator_, mask_f, num_count_),
                                 "Failed to generate uniform");
    DropoutForward(input, mask, output, mask_f, num_count_, keep_prob_, reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    InitResource();

    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 1, but got " << input_num;
    }

    auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }

    num_count_ = 1;
    for (size_t x : input_shape) {
      num_count_ *= x;
    }
    keep_prob_ = GetAttr<float>(kernel_node, "keep_prob");
    if (!states_init_) {
      int64_t seed = GetAttr<int64_t>(kernel_node, "Seed0");
      if (seed == 0) {
        seed = GetAttr<int64_t>(kernel_node, "Seed1");
        if (seed == 0) {
          seed = time(NULL);
        }
      }
      seed_ = static_cast<uint64_t>(seed);
      CHECK_CURAND_RET_WITH_EXCEPT(curandCreateGenerator(&mask_generator_, CURAND_RNG_PSEUDO_DEFAULT),
                                   "Failed to create generator");
      CHECK_CURAND_RET_WITH_EXCEPT(curandSetPseudoRandomGeneratorSeed(mask_generator_, seed_),
                                   "Failed to SetPseudoRandomGeneratorSeed");
      MS_EXCEPTION_IF_NULL(mask_generator_);
      states_init_ = true;
    }

    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    cudnn_handle_ = nullptr;
    is_null_input_ = false;
    kernel_name_ = "Dropout";
    num_count_ = 0;
    keep_prob_ = 0.0;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitResource() override { cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle(); }

  void InitSizeLists() override {
    size_t input_size = num_count_ * sizeof(T);
    input_size_list_.push_back(input_size);
    output_size_list_.push_back(input_size);                     // output size: the same with input size
    output_size_list_.push_back(input_size);                     // mask size: the same with input size
    workspace_size_list_.push_back(num_count_ * sizeof(float));  // temp mask_f for curandGen
  }

 private:
  cudnnHandle_t cudnn_handle_;
  bool is_null_input_;
  std::string kernel_name_;
  size_t num_count_;
  float keep_prob_;
  bool states_init_{false};
  uint64_t seed_{0};
  curandGenerator_t mask_generator_{nullptr};
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_DROPOUT_GPU_KERNEL_H_
