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
#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_MESHGRID_GPU_KERNEL_H
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_MESHGRID_GPU_KERNEL_H

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/oneslike_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/math/broadcast_gpu_kernel.h"

namespace luojianet_ms {
namespace kernel {
template <typename T>
class MeshgridGpuKernelMod : public NativeGpuKernelMod {
 public:
  MeshgridGpuKernelMod() { ResetResource(); }
  ~MeshgridGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *ones_device = GetDeviceAddress<T>(workspace, 0);
    CalOnesLike(output_size_, static_cast<T *>(nullptr), ones_device, reinterpret_cast<cudaStream_t>(stream_ptr));

    std::vector<size_t> broadcasted_ones_shape(MAX_DIMS, 1);
    for (size_t i = 0; i < output_shape_.size(); i++) {
      broadcasted_ones_shape[i] = output_shape_[i];
    }

    for (size_t i = 0; i < outputs.size(); i++) {
      T *input_device = GetDeviceAddress<T>(inputs, i);
      T *output_device = GetDeviceAddress<T>(outputs, i);
      std::vector<size_t> broadcasted_input_shape(MAX_DIMS, 1);
      broadcasted_input_shape[i] = input_shapes_[i];

      if (swap_indexing_ && i < 2) {
        std::swap(broadcasted_input_shape[0], broadcasted_input_shape[1]);
      }

      BroadcastArith(broadcasted_input_shape, broadcasted_ones_shape, output_shape_, BROADCAST_TYPE_MUL, input_device,
                     ones_device, output_device, reinterpret_cast<cudaStream_t>(stream_ptr));
    }

    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    std::string indexing = GetAttr<std::string>(kernel_node, "indexing");
    if (indexing == "xy") {
      swap_indexing_ = true;
    } else if (indexing == "ij") {
      swap_indexing_ = false;
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the value of 'indexing' should be \"xy\" or \"ij\", but got "
                        << indexing;
    }

    input_size_ = 1;
    input_count_ = common::AnfAlgo::GetInputTensorNum(kernel_node);
    for (size_t i = 0; i < input_count_; i++) {
      auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, i);
      if (input_shape.size() < 1) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of input[" << i << "] cannot be less than 1, "
                          << "but got " << input_shape.size();
      }
      size_t input_size = input_shape[0];
      input_shapes_.push_back(input_size);
      input_size_ *= input_size;
    }

    output_size_ = 1;
    output_count_ = common::AnfAlgo::GetOutputTensorNum(kernel_node);

    // inferred shape swaps output shape for us if needed
    output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(output_shape_, kernel_name, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }

    if (output_count_ != input_count_) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name
                        << "', the number of inputs and outputs should be the same, but got the number of inputs: "
                        << input_count_ << ", the number of outputs: " << output_count_;
    }

    for (size_t i = 0; i < output_shape_.size(); i++) {
      output_size_ *= output_shape_[i];
    }

    // need to pad output shape with ones for broadcast kernel
    int need_broadcast_size = MAX_DIMS - output_shape_.size();
    for (int i = 0; i < need_broadcast_size; i++) {
      output_shape_.push_back(1);
    }

    InitSizeLists();

    return true;
  }

  void ResetResource() noexcept override {
    input_shapes_.clear();
    output_shape_.clear();
    input_size_ = 0;
    input_count_ = 0;
    output_size_ = 0;
    output_count_ = 0;
    swap_indexing_ = true;
    is_null_input_ = false;

    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    for (const size_t &input_shape : input_shapes_) {
      input_size_list_.push_back(input_shape * sizeof(T));
    }

    for (size_t i = 0; i < output_count_; i++) {
      output_size_list_.push_back(output_size_ * sizeof(T));
    }

    workspace_size_list_.push_back(output_size_ * sizeof(T));
  }

 private:
  std::vector<size_t> input_shapes_;
  std::vector<size_t> output_shape_;
  size_t input_size_;
  size_t input_count_;
  size_t output_size_;
  size_t output_count_;
  bool swap_indexing_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_MESHGRID_GPU_KERNEL_H
