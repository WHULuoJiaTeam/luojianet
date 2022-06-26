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

#include "plugin/device/cpu/kernel/argmax_cpu_kernel.h"

#include <string>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kArgMaxInputsNum = 1;
constexpr size_t kArgMaxOutputsNum = 1;
constexpr char kKernelName[] = "ArgMax";

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
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kArgMaxInputsNum, kKernelName);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kArgMaxOutputsNum, kKernelName);
  size_t data_size = sizeof(T);
  size_t input_size = get_element_num(shape) * data_size;
  size_t output_num = num_before_axis * num_after_axis;
  size_t output_size = output_num * sizeof(int);
  if (inputs[0]->size != input_size) {
    MS_LOG(EXCEPTION) << "For '" << kKernelName << "', the memory size of 'input_x' should be equal to " << input_size
                      << ", but got the memory size is " << inputs[0]->size;
  }
  if (outputs[0]->size != output_size) {
    MS_LOG(EXCEPTION) << "For '" << kKernelName << "', the memory size of output should be equal to " << output_size
                      << ", but got the memory size is " << outputs[0]->size;
  }
  return true;
}
}  // namespace

template <typename T>
bool ArgmaxCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  if (!check_validation<T>(shape_, num_before_axis_, num_after_axis_, inputs, outputs)) {
    return false;
  }

  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output = reinterpret_cast<int32_t *>(outputs[0]->addr);

  std::vector<float> array_axis(dim_axis_);
  for (size_t i = 0; i < num_before_axis_; i++) {
    size_t src_index_i = i * dim_axis_ * num_after_axis_;
    for (size_t j = 0; j < num_after_axis_; j++) {
      size_t src_index_j = src_index_i + j;
      for (size_t k = 0; k < dim_axis_; k++) {
        size_t src_index_k = k * num_after_axis_ + src_index_j;
        array_axis[k] = static_cast<float>(input[src_index_k]);
      }
      auto max_ops = std::max_element(array_axis.begin(), array_axis.end());
      auto max_index = static_cast<int32_t>(std::distance(array_axis.begin(), max_ops));
      auto dst_index = i * num_after_axis_ + j;
      output[dst_index] = max_index;
    }
  }
  return true;
}

void ArgmaxCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  size_t shape_len = shape_.size();
  if (shape_len == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'input_x' should be at least 1, but got 0.";
  }
  int64_t axis = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS);
  axis += SizeToLong(shape_len);
  if (axis < 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' should be in range [-1, " << (shape_len - 1)
                      << "], but got " << axis;
  }
  axis = axis % SizeToLong(shape_len);
  num_before_axis_ = 1;
  num_after_axis_ = 1;
  for (size_t i = 0; i < shape_len; i++) {
    if (SizeToLong(i) < axis) {
      num_before_axis_ *= shape_[i];
    } else if (SizeToLong(i) > axis) {
      num_after_axis_ *= shape_[i];
    }
  }
  dim_axis_ = shape_[LongToSize(axis)];

  auto build_info = AnfAlgo::GetSelectKernelBuildInfo(kernel_node);
  if (build_info->GetInputNum() < 1) {
    MS_LOG(EXCEPTION) << "Argmax input size should not less than 1!";
  }
  auto input_type_id = build_info->GetInputDeviceType(0);
  switch (input_type_id) {
    case kNumberTypeFloat32:
      kernel_func_ = &ArgmaxCpuKernelMod::LaunchKernel<float>;
      break;
    case kNumberTypeFloat16:
      kernel_func_ = &ArgmaxCpuKernelMod::LaunchKernel<float16>;
      break;
    default:
      MS_LOG(EXCEPTION) << "Argmax kernel does not support " << TypeIdToString(input_type_id);
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Argmax, ArgmaxCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
