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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_SQUARED_DIFFERENCE_GPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_SQUARED_DIFFERENCE_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_impl.cuh"
#include "plugin/device/gpu/kernel/kernel_constants.h"
namespace luojianet_ms {
namespace kernel {
constexpr int MAX_DIMS = 7;
template <typename T>
class SquaredDifferenceOpGpuKernelMod : public NativeGpuKernelMod {
 public:
  SquaredDifferenceOpGpuKernelMod() { ResetResource(); }
  ~SquaredDifferenceOpGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *lhs = GetDeviceAddress<T>(inputs, 0);
    T *rhs = GetDeviceAddress<T>(inputs, 1);
    T *output = GetDeviceAddress<T>(outputs, 0);
    if (need_broadcast_) {
      BroadcastArith(lhs_shape_, rhs_shape_, output_shape_, op_type_, lhs, rhs, output,
                     reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      ElewiseArith(output_num_, op_type_, lhs, rhs, output, reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    auto input_shape1 = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 0);
    auto input_shape2 = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 1);
    auto output_shape = AnfAlgo::GetOutputDeviceShapeAdaptively(kernel_node, 0);
    kernel_node_ = kernel_node;
    is_null_input_ = CHECK_SHAPE_NULL(input_shape1, kernel_name, "input") ||
                     CHECK_SHAPE_NULL(input_shape2, kernel_name, "input") ||
                     CHECK_SHAPE_NULL(output_shape, kernel_name, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    need_broadcast_ = common::AnfAlgo::IsTensorBroadcast(input_shape1, input_shape2);
    if (need_broadcast_ && output_shape.size() > MAX_DIMS) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of output cannot be greater than " << MAX_DIMS
                        << ", but got " << output_shape.size();
    }

    lhs_shape_.resize(MAX_DIMS, 1);
    rhs_shape_.resize(MAX_DIMS, 1);
    output_shape_.resize(MAX_DIMS, 1);
    for (size_t i = 0; i < output_shape.size(); i++) {
      if (need_broadcast_) {
        output_shape_[i] = output_shape[i];
      }
      output_num_ *= output_shape[i];
    }
    int lhs_offset = output_shape.size() - input_shape1.size();
    for (size_t j = 0; j < input_shape1.size(); j++) {
      if (need_broadcast_) {
        if ((j + lhs_offset) >= 0 && (j + lhs_offset) < MAX_DIMS) {
          lhs_shape_[j + lhs_offset] = input_shape1[j];
        } else {
          auto index = j + lhs_offset;
          MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the index of input cannot be " << index << ", but got "
                            << index;
        }
      }
      input1_num_ *= input_shape1[j];
    }
    int rhs_offset = output_shape.size() - input_shape2.size();
    for (size_t k = 0; k < input_shape2.size(); k++) {
      if (need_broadcast_) {
        if ((k + rhs_offset) >= 0 && (k + rhs_offset) < MAX_DIMS) {
          rhs_shape_[k + rhs_offset] = input_shape2[k];
        } else {
          auto index = k + rhs_offset;
          MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the index of input cannot be " << index << ", but got "
                            << index;
        }
      }
      input2_num_ *= input_shape2[k];
    }

    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    op_type_ = BROADCAST_TYPE_SQUARED_DIFFERENCE;
    need_broadcast_ = false;
    is_comp_op_ = false;
    is_null_input_ = false;
    input1_num_ = 1;
    input2_num_ = 1;
    output_num_ = 1;
    lhs_shape_.clear();
    rhs_shape_.clear();
    output_shape_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitResource() override { return; }
  void InitSizeLists() override {
    input_size_list_.push_back(input1_num_ * sizeof(T));
    input_size_list_.push_back(input2_num_ * sizeof(T));
    output_size_list_.push_back(output_num_ * sizeof(T));
  }

 private:
  BroadcastOpType op_type_;
  bool need_broadcast_;
  bool is_comp_op_;
  bool is_null_input_;
  size_t input1_num_;
  size_t input2_num_;
  size_t output_num_;
  std::vector<size_t> lhs_shape_;
  std::vector<size_t> rhs_shape_;
  std::vector<size_t> output_shape_;
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_SQUARED_DIFFERENCE_GPU_KERNEL_H_
