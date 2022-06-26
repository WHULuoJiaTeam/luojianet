/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/argmin_with_value_cpu_kernel.h"

#include <string>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kArgMinWithValueInputsNum = 1;
constexpr size_t kArgMinWithValueOutputsNum = 2;
constexpr char kKernelName[] = "ArgMinWithValue";

size_t get_element_num(const std::vector<size_t> &shape) {
  size_t size = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }
  return size;
}

template <typename T>
bool check_validation(const std::vector<size_t> &shape, const size_t num_before_axis, const size_t num_after_axis,
                      const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kArgMinWithValueInputsNum, kKernelName);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kArgMinWithValueOutputsNum, kKernelName);
  size_t data_size = sizeof(T);
  size_t input_size = get_element_num(shape) * data_size;
  size_t output_num = num_before_axis * num_after_axis;
  size_t out0_size = output_num * sizeof(int);
  size_t out1_size = output_num * data_size;
  if (inputs[0]->size != input_size) {
    MS_LOG(EXCEPTION) << "For '" << kKernelName << "', the memory size of 'input_x' should be " << input_size
                      << ", but got the memory size is " << inputs[0]->size;
  }
  if (outputs[0]->size != out0_size) {
    MS_LOG(EXCEPTION) << "For '" << kKernelName << "', the memory size of the 1st output should be " << out0_size
                      << ", but got the memory size is " << outputs[0]->size;
  }
  if (outputs[1]->size != out1_size) {
    MS_LOG(EXCEPTION) << "For '" << kKernelName << "', the memory size of the 2nd output should be " << out1_size
                      << ", but got the memory size is " << outputs[1]->size;
  }
  return true;
}
}  // namespace

template <typename T>
bool ArgMinWithValueCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                               const std::vector<kernel::AddressPtr> &,
                                               const std::vector<kernel::AddressPtr> &outputs) {
  if (!check_validation<T>(shape_, num_before_axis_, num_after_axis_, inputs, outputs)) {
    return false;
  }

  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output0 = reinterpret_cast<int32_t *>(outputs[0]->addr);
  auto *output1 = reinterpret_cast<T *>(outputs[1]->addr);
  std::vector<float> array_axis(dim_axis_);
  for (size_t i = 0; i < num_before_axis_; i++) {
    size_t src_index_i = i * dim_axis_ * num_after_axis_;
    for (size_t j = 0; j < num_after_axis_; j++) {
      size_t src_index_j = src_index_i + j;
      for (size_t k = 0; k < dim_axis_; k++) {
        size_t src_index_k = k * num_after_axis_ + src_index_j;
        array_axis[k] = static_cast<float>(input[src_index_k]);
      }
      auto min_ops = std::min_element(array_axis.begin(), array_axis.end());
      auto min_index = static_cast<int32_t>(std::distance(array_axis.begin(), min_ops));
      size_t dst_index = i * num_after_axis_ + j;
      output0[dst_index] = min_index;
      size_t src_index = IntToSize(min_index) * num_after_axis_ + src_index_j;
      output1[dst_index] = input[src_index];
    }
  }
  return true;
}

void ArgMinWithValueCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  size_t shape_len = shape_.size();
  if (shape_len == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'input_x' should be at least 1, but got 0.";
  }
  int64_t axis = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS);
  axis += static_cast<int64_t>(shape_len);
  if (axis < 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' should be in range [-1, " << (shape_len - 1)
                      << "], but got " << axis;
  }
  axis = axis % static_cast<int64_t>(shape_len);
  num_before_axis_ = 1;
  num_after_axis_ = 1;
  for (size_t i = 0; i < shape_len; i++) {
    if (static_cast<int64_t>(i) < axis) {
      num_before_axis_ *= shape_[i];
    } else if (static_cast<int64_t>(i) > axis) {
      num_after_axis_ *= shape_[i];
    }
  }
  dim_axis_ = shape_[axis];

  auto build_info = AnfAlgo::GetSelectKernelBuildInfo(kernel_node);
  if (build_info->GetInputNum() < 1) {
    MS_LOG(EXCEPTION) << "Argmax input size should not less than 1!";
  }
  auto input_type_id = build_info->GetInputDeviceType(0);
  switch (input_type_id) {
    case kNumberTypeFloat32:
      kernel_func_ = &ArgMinWithValueCpuKernelMod::LaunchKernel<float>;
      break;
    case kNumberTypeFloat16:
      kernel_func_ = &ArgMinWithValueCpuKernelMod::LaunchKernel<float16>;
      break;
    default:
      MS_LOG(EXCEPTION) << "Argmax kernel does not support " << TypeIdToString(input_type_id);
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ArgMinWithValue, ArgMinWithValueCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
