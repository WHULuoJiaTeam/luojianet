/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/maximum_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kShapeIndexZero = 0;
constexpr auto kShapeIndex1st = 1;
constexpr auto kShapeIndex2nd = 2;
constexpr auto kShapeIndex3rd = 3;
constexpr auto kShapeIndex4th = 4;
constexpr auto kShapeIndex5th = 5;
constexpr auto kShapeIndex6th = 6;
constexpr size_t kMaximumInputsNum = 2;
constexpr size_t kMaximumOutputsNum = 1;
}  // namespace

void MaximumCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_x_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  input_y_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  TypeId input_x_dtype = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  TypeId input_y_dtype = AnfAlgo::GetInputDeviceDataType(kernel_node, 1);
  size_t max_input_shape_size =
    input_x_shape_.size() > input_y_shape_.size() ? input_x_shape_.size() : input_y_shape_.size();
  for (size_t i = 0; i < output_shape_.size(); i++) {
    output_num_ *= output_shape_[i];
  }
  if ((input_x_shape_.size() == 0 && input_y_shape_.size() != 0) ||
      (input_x_shape_.size() != 0 && input_y_shape_.size() == 0)) {
    InitInputTensorAndScalar(max_input_shape_size);
  } else if (max_input_shape_size == output_shape_.size() && output_shape_.size() != 0) {
    InitInputTensors(input_x_dtype, input_y_dtype);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', inputs should be two tensors or one tensor and one scalar, but got " << input_x_dtype
                      << " and " << input_y_dtype;
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaximumLaunchFunc> &pair) { return pair.first; });
  auto [is_match, index] = MatchKernelAttr(kernel_attr, support_list);
  if (!is_match) {
    MS_LOG(EXCEPTION) << "Maximum does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

void MaximumCpuKernelMod::InitInputTensorAndScalar(size_t max_input_shape_size) {
  if (max_input_shape_size != output_shape_.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of output tensor should be equal to the max "
                         "dimension of inputs, but got the dimension of output tensor: "
                      << output_shape_.size() << " and the max dimension of inputs: " << max_input_shape_size;
  }
  need_broadcast_ = false;
}

void MaximumCpuKernelMod::InitInputTensors(TypeId input_x_dtype, TypeId input_y_dtype) {
  if (input_x_dtype == kNumberTypeBool && input_y_dtype == kNumberTypeBool) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', input tensor types should not be both bool.";
  }
  // Check if the shape needs to be broadcast
  need_broadcast_ = IsBroadcast();
  if (need_broadcast_) {
    InitTensorBroadcastShape();
  }
}

template <typename T>
bool MaximumCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaximumInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaximumOutputsNum, kernel_name_);
  T *input_x_ = reinterpret_cast<T *>(inputs[0]->addr);
  T *input_y_ = reinterpret_cast<T *>(inputs[1]->addr);
  T *output_ = reinterpret_cast<T *>(outputs[0]->addr);
  BroadcastArith(input_x_, input_y_, output_);
  return true;
}

template <typename T>
void MaximumCpuKernelMod::BroadcastArith(const T *input_x, const T *input_y, T *output) const {
  MS_EXCEPTION_IF_NULL(input_x);
  MS_EXCEPTION_IF_NULL(input_y);
  MS_EXCEPTION_IF_NULL(output);
  if (need_broadcast_) {
    BroadcastArithKernel(broadcast_input_x_shape_[kShapeIndexZero], broadcast_input_x_shape_[kShapeIndex1st],
                         broadcast_input_x_shape_[kShapeIndex2nd], broadcast_input_x_shape_[kShapeIndex3rd],
                         broadcast_input_x_shape_[kShapeIndex4th], broadcast_input_x_shape_[kShapeIndex5th],
                         broadcast_input_x_shape_[kShapeIndex6th], broadcast_input_y_shape_[kShapeIndexZero],
                         broadcast_input_y_shape_[kShapeIndex1st], broadcast_input_y_shape_[kShapeIndex2nd],
                         broadcast_input_y_shape_[kShapeIndex3rd], broadcast_input_y_shape_[kShapeIndex4th],
                         broadcast_input_y_shape_[kShapeIndex5th], broadcast_input_y_shape_[kShapeIndex6th],
                         broadcast_output_shape_[kShapeIndexZero], broadcast_output_shape_[kShapeIndex1st],
                         broadcast_output_shape_[kShapeIndex2nd], broadcast_output_shape_[kShapeIndex3rd],
                         broadcast_output_shape_[kShapeIndex4th], broadcast_output_shape_[kShapeIndex5th],
                         broadcast_output_shape_[kShapeIndex6th], input_x, input_y, output);
  } else {
    if (input_x_shape_.size() == 0 || input_y_shape_.size() == 0) {
      BroadcastArithOneScalarOneTensor(input_x, input_y, output);
    } else {
      BroadcastArithTensors(input_x, input_y, output);
    }
  }
}

bool MaximumCpuKernelMod::IsBroadcast() const {
  if (input_x_shape_.size() != input_y_shape_.size()) {
    return true;
  }
  for (size_t i = 0; i < input_x_shape_.size(); i++) {
    if (input_x_shape_[i] != input_y_shape_[i]) {
      return true;
    }
  }
  return false;
}

void MaximumCpuKernelMod::InitTensorBroadcastShape() {
  if (output_shape_.size() > max_dims_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of output should be less than or equal to 7, but got "
                      << output_shape_.size() << ".";
  }
  broadcast_input_x_shape_.resize(max_dims_, 1);
  broadcast_input_y_shape_.resize(max_dims_, 1);
  broadcast_output_shape_.resize(max_dims_, 1);
  for (size_t i = 0; i < output_shape_.size(); i++) {
    broadcast_output_shape_[i] = output_shape_[i];
  }
  int input_x_dim_offset = output_shape_.size() - input_x_shape_.size();
  for (size_t j = 0; j < input_x_shape_.size(); j++) {
    broadcast_input_x_shape_[j + IntToSize(input_x_dim_offset)] = input_x_shape_[j];
    input_x_num_ *= input_x_shape_[j];
  }
  int input_y_dim_offset = output_shape_.size() - input_y_shape_.size();
  for (size_t k = 0; k < input_y_shape_.size(); k++) {
    if (need_broadcast_) {
      broadcast_input_y_shape_[k + IntToSize(input_y_dim_offset)] = input_y_shape_[k];
      input_y_num_ *= input_y_shape_[k];
    }
  }
}

// Broadcast comparison
size_t MaximumCpuKernelMod::Index(const size_t &index, const size_t &dim) const { return dim == 1 ? 0 : index; }

// Broadcast Arithmetic
template <typename T>
void MaximumCpuKernelMod::BroadcastArithKernel(const size_t l0, const size_t l1, const size_t l2, const size_t l3,
                                               const size_t l4, const size_t l5, const size_t l6, const size_t r0,
                                               const size_t r1, const size_t r2, const size_t r3, const size_t r4,
                                               const size_t r5, const size_t r6, const size_t d0, const size_t d1,
                                               const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                               const size_t d6, const T *input_x, const T *input_y, T *output) const {
  for (size_t pos = 0; pos < output_num_; pos++) {
    size_t i = pos / (d1 * d2 * d3 * d4 * d5 * d6) % d0;
    size_t j = pos / (d2 * d3 * d4 * d5 * d6) % d1;
    size_t k = pos / (d3 * d4 * d5 * d6) % d2;
    size_t l = pos / (d4 * d5 * d6) % d3;
    size_t m = pos / (d5 * d6) % d4;
    size_t n = pos / d6 % d5;
    size_t o = pos % d6;

    size_t l_index = Index(i, l0) * l1 * l2 * l3 * l4 * l5 * l6;
    l_index += Index(j, l1) * l2 * l3 * l4 * l5 * l6;
    l_index += Index(k, l2) * l3 * l4 * l5 * l6;
    l_index += Index(l, l3) * l4 * l5 * l6;
    l_index += Index(m, l4) * l5 * l6;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    size_t r_index = Index(i, r0) * r1 * r2 * r3 * r4 * r5 * r6;
    r_index += Index(j, r1) * r2 * r3 * r4 * r5 * r6;
    r_index += Index(k, r2) * r3 * r4 * r5 * r6;
    r_index += Index(l, r3) * r4 * r5 * r6;
    r_index += Index(m, r4) * r5 * r6;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    output[pos] = MaximumFunc(input_x[l_index], input_y[r_index]);
  }
}

template <typename T>
void MaximumCpuKernelMod::BroadcastArithOneScalarOneTensor(const T *input_x, const T *input_y, T *output) const {
  if (input_x_shape_.size() == 0) {
    for (size_t i = 0; i < output_num_; ++i) {
      output[i] = MaximumFunc(input_x[0], input_y[i]);
    }
  } else {
    for (size_t i = 0; i < output_num_; ++i) {
      output[i] = MaximumFunc(input_x[i], input_y[0]);
    }
  }
}

template <typename T>
void MaximumCpuKernelMod::BroadcastArithTensors(const T *input_x, const T *input_y, T *output) const {
  for (size_t i = 0; i < output_num_; ++i) {
    output[i] = MaximumFunc(input_x[i], input_y[i]);
  }
}

std::vector<std::pair<KernelAttr, MaximumCpuKernelMod::MaximumLaunchFunc>> MaximumCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &MaximumCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &MaximumCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
   &MaximumCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
   &MaximumCpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &MaximumCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &MaximumCpuKernelMod::LaunchKernel<double>}};

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Maximum, MaximumCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
