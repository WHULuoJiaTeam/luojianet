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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SCATTER_FUNCTOR_GPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SCATTER_FUNCTOR_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/scatter_functor_impl.cuh"

namespace luojianet_ms {
namespace kernel {

static const std::map<std::string, ScatterFunctorType> kScatterFunctorTypeMap = {
  {"ScatterUpdate", SCATTER_FUNC_UPDATE},
  {"ScatterAdd", SCATTER_FUNC_ADD},
  {"ScatterSub", SCATTER_FUNC_SUB},
};

template <typename T, typename S>
class ScatterFunctorKernelMod : public NativeGpuKernelMod {
 public:
  ScatterFunctorKernelMod() { ResetResource(); }
  ~ScatterFunctorKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input = GetDeviceAddress<T>(inputs, 0);
    S *indices = GetDeviceAddress<S>(inputs, 1);
    T *updates = GetDeviceAddress<T>(inputs, 2);
    T *output = GetDeviceAddress<T>(outputs, 0);

    ScatterFunc(scatter_functor_type_, inner_size_, indices_size_, indices, updates, input,
                reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(&output[0], &input[0], input_size_ * sizeof(T), cudaMemcpyDeviceToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync output failed");
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    auto iter = kScatterFunctorTypeMap.find(kernel_name);
    if (iter == kScatterFunctorTypeMap.end()) {
      MS_LOG(EXCEPTION)
        << "For '" << kernel_name
        << "Only support these scatter functors: ScatterUpdate, ScatterAdd or ScatterSub currently, but got "
        << kernel_name;
    } else {
      scatter_functor_type_ = iter->second;
    }
    kernel_node_ = kernel_node;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 3) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 3, but got " << input_num;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 1, but got " << output_num;
    }
    auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    input_size_ = 1;
    inner_size_ = 1;
    for (size_t i = 1; i < input_shape.size(); i++) {
      inner_size_ *= input_shape[i];
    }
    input_size_ = input_shape[0] * inner_size_;
    auto indices_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    indices_size_ = 1;
    for (size_t i = 0; i < indices_shape.size(); i++) {
      indices_size_ *= indices_shape[i];
    }
    updates_size_ = indices_size_ * inner_size_;
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = 0;
    inner_size_ = 0;
    indices_size_ = 0;
    updates_size_ = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    input_size_list_.push_back(indices_size_ * sizeof(S));
    input_size_list_.push_back(updates_size_ * sizeof(T));
    output_size_list_.push_back(input_size_ * sizeof(T));
  }

 private:
  ScatterFunctorType scatter_functor_type_;
  size_t input_size_;
  size_t inner_size_;
  size_t indices_size_;
  size_t updates_size_;
};
}  // namespace kernel
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SCATTER_FUNCTOR_GPU_KERNEL_H_
