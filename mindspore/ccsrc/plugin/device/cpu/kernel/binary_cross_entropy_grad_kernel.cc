/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/binary_cross_entropy_grad_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBceGradInputsNumWithWeight = 4;
constexpr size_t kBceGradOutputsNum = 1;
}  // namespace

template <typename T>
void BinaryCrossEntropyGradCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                      const std::vector<AddressPtr> &outputs) const {
  const auto *input_x = reinterpret_cast<T *>(inputs[0]->addr);
  const auto *input_y = reinterpret_cast<T *>(inputs[1]->addr);
  const auto *dloss = reinterpret_cast<T *>(inputs[2]->addr);
  const T *weight = weight_defined_ ? reinterpret_cast<T *>(inputs[3]->addr) : nullptr;
  auto *dx = reinterpret_cast<T *>(outputs[0]->addr);
  auto epsilon = static_cast<T>(1e-12);
  auto one = static_cast<T>(1);

  if (reduction_ == kNone) {
    if (weight_defined_) {
      for (size_t i = 0; i < input_size_; i++) {
        T denominator = ((input_x[i] * (one - input_x[i])) > epsilon) ? (input_x[i] * (one - input_x[i])) : epsilon;
        T value = weight[i] * (input_x[i] - input_y[i]) / denominator;
        dx[i] = value * dloss[i];
      }
    } else {
      for (size_t i = 0; i < input_size_; i++) {
        T denominator = ((input_x[i] * (one - input_x[i])) > epsilon) ? (input_x[i] * (one - input_x[i])) : epsilon;
        T value = (input_x[i] - input_y[i]) / denominator;
        dx[i] = value * dloss[i];
      }
    }
  } else {
    T dloss1 = dloss[0];
    if (reduction_ == kMean) {
      dloss1 = dloss[0] / static_cast<T>(input_size_);
    }
    if (weight_defined_) {
      for (size_t i = 0; i < input_size_; i++) {
        T denominator = ((input_x[i] * (one - input_x[i])) > epsilon) ? (input_x[i] * (one - input_x[i])) : epsilon;
        T value = weight[i] * (input_x[i] - input_y[i]) / denominator;
        dx[i] = value * dloss1;
      }
    } else {
      for (size_t i = 0; i < input_size_; i++) {
        T denominator = ((input_x[i] * (one - input_x[i])) > epsilon) ? (input_x[i] * (one - input_x[i])) : epsilon;
        T value = (input_x[i] - input_y[i]) / denominator;
        dx[i] = value * dloss1;
      }
    }
  }
}

bool BinaryCrossEntropyGradCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                                const std::vector<AddressPtr> &outputs) {
  const size_t expect_inputs_num = weight_defined_ ? kBceGradInputsNumWithWeight : kBceGradInputsNumWithWeight - 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), expect_inputs_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kBceGradOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of input should be float16 or float32, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  return true;
}

void BinaryCrossEntropyGradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  weight_defined_ = (input_num == kBceGradInputsNumWithWeight);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  for (size_t i = 0; i < input_shape.size(); i++) {
    input_size_ *= input_shape[i];
  }

  const std::string reduction = common::AnfAlgo::GetNodeAttr<string>(kernel_node, REDUCTION);
  if (reduction == NONE) {
    reduction_ = kNone;
  } else if (reduction == MEAN) {
    reduction_ = kMean;
  } else if (reduction == SUM) {
    reduction_ = kSum;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'reduction' should be 'none', 'mean', or 'sum', but got "
                      << reduction;
  }
}

std::vector<KernelAttr> BinaryCrossEntropyGradCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddOutputAttr(kNumberTypeFloat16),
                                                     KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddOutputAttr(kNumberTypeFloat32),
                                                     KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddOutputAttr(kNumberTypeFloat16),
                                                     KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddOutputAttr(kNumberTypeFloat32)};

  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BinaryCrossEntropyGrad, BinaryCrossEntropyGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
