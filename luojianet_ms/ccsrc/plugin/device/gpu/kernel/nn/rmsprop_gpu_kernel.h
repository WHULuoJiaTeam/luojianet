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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_RMSPROP_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_RMSPROP_KERNEL_H_

#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/rmsprop_impl.cuh"

namespace luojianet_ms {
namespace kernel {
template <typename T>
class RMSPropGpuKernelMod : public NativeGpuKernelMod {
 public:
  RMSPropGpuKernelMod()
      : size_(1), use_center_(false), is_null_input_(false), decay_(0.0), momentum_(0.9), epsilon_(1e-12) {}
  ~RMSPropGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream) override {
    if (is_null_input_) {
      return true;
    }
    if (!use_center_) {
      T *variable = GetDeviceAddress<T>(inputs, 0);
      T *mean_square = GetDeviceAddress<T>(inputs, 1);
      T *moment = GetDeviceAddress<T>(inputs, 2);
      T *learning_rate = GetDeviceAddress<T>(inputs, 3);
      T *gradients = GetDeviceAddress<T>(inputs, 4);

      RmsProp(learning_rate, decay_, momentum_, epsilon_, variable, mean_square, moment, gradients, size_,
              reinterpret_cast<cudaStream_t>(stream));
    } else {
      T *variable = GetDeviceAddress<T>(inputs, 0);
      T *mean_gradients = GetDeviceAddress<T>(inputs, 1);
      T *mean_square = GetDeviceAddress<T>(inputs, 2);
      T *moment = GetDeviceAddress<T>(inputs, 3);
      T *gradients = GetDeviceAddress<T>(inputs, 4);
      T *learning_rate = GetDeviceAddress<T>(inputs, 5);
      T *decay = GetDeviceAddress<T>(inputs, 6);
      T *momentum = GetDeviceAddress<T>(inputs, 7);
      T *epsilon = GetDeviceAddress<T>(inputs, 8);

      RmsPropCenter(learning_rate, decay, momentum, epsilon, variable, mean_gradients, mean_square, moment, gradients,
                    size_, reinterpret_cast<cudaStream_t>(stream));
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    auto node_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    if (node_name == "ApplyCenteredRMSProp") {
      use_center_ = true;
    }

    if (node_name == "ApplyRMSProp") {
      decay_ = GetAttr<float>(kernel_node, "rho");
      momentum_ = GetAttr<float>(kernel_node, "momentum");
      epsilon_ = GetAttr<float>(kernel_node, "epsilon");
    }
    auto input_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, node_name, "var");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    for (auto &dim : input_shape) {
      size_ *= dim;
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    size_t input_size = size_ * sizeof(T);
    if (!use_center_) {
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(sizeof(T));
      input_size_list_.push_back(input_size);
      output_size_list_.push_back(input_size);
    } else {
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(sizeof(T));
      input_size_list_.push_back(sizeof(T));
      input_size_list_.push_back(sizeof(T));
      input_size_list_.push_back(sizeof(T));
      output_size_list_.push_back(input_size);
    }
  }

 private:
  size_t size_;
  bool use_center_;
  bool is_null_input_;
  float decay_;
  float momentum_;
  float epsilon_;
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif
