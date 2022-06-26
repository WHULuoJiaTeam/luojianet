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

#include "plugin/device/cpu/kernel/binary_cross_entropy_cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBceInputsNumWithWeight = 3;
constexpr size_t kBceOutputsNum = 1;
}  // namespace

template <typename T>
void BinaryCrossEntropyCpuKernelMod::LaunchToScalar(const int &input_size, const ReductionType &reduction, T *loss,
                                                    T *tmp_loss) const {
  if (input_size % 2 == 1) {
    tmp_loss[0] += tmp_loss[input_size - 1];
  }

  for (int stride = input_size / 2; stride > 0; stride = stride / 2) {
    for (int i = 0; i < stride; i++) {
      tmp_loss[i] += tmp_loss[i + stride];
    }
    if (stride > 2 && stride % 2 == 1) {
      tmp_loss[0] += tmp_loss[stride - 1];
    }
  }

  loss[0] = tmp_loss[0];
  if (reduction == kMean) {
    loss[0] /= static_cast<T>(input_size);
  }
}

template <typename T>
void BinaryCrossEntropyCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                  const std::vector<AddressPtr> &outputs) const {
  const auto *input_x = reinterpret_cast<T *>(inputs[0]->addr);
  const auto *input_y = reinterpret_cast<T *>(inputs[1]->addr);
  const T *weight = weight_defined_ ? reinterpret_cast<T *>(inputs[2]->addr) : nullptr;
  auto *loss = reinterpret_cast<T *>(outputs[0]->addr);
  std::vector<T> tmp_loss(input_size_);
  auto epsilon = static_cast<T>(1e-12);
  auto one = static_cast<T>(1);

  if (reduction_ == kNone) {
    if (weight_defined_) {
      for (size_t i = 0; i < input_size_; i++) {
        auto value = static_cast<T>(
          -weight[i] * (input_y[i] * log(input_x[i] + epsilon) + (one - input_y[i]) * log(one - input_x[i] + epsilon)));
        loss[i] = value;
      }
    } else {
      for (size_t i = 0; i < input_size_; i++) {
        auto value = static_cast<T>(
          -(input_y[i] * log(input_x[i] + epsilon) + (one - input_y[i]) * log(one - input_x[i] + epsilon)));
        loss[i] = value;
      }
    }
  } else {
    if (weight_defined_) {
      for (size_t i = 0; i < input_size_; i++) {
        auto value = static_cast<T>(
          -weight[i] * (input_y[i] * log(input_x[i] + epsilon) + (one - input_y[i]) * log(one - input_x[i] + epsilon)));
        tmp_loss[i] = value;
      }
    } else {
      for (size_t i = 0; i < input_size_; i++) {
        auto value = static_cast<T>(
          -(input_y[i] * log(input_x[i] + epsilon) + (one - input_y[i]) * log(one - input_x[i] + epsilon)));
        tmp_loss[i] = value;
      }
    }
  }
  if (reduction_ != kNone) {
    LaunchToScalar<T>(input_size_, reduction_, loss, tmp_loss.data());
  }
}

bool BinaryCrossEntropyCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                            const std::vector<AddressPtr> &outputs) {
  const size_t expect_inputs_num = weight_defined_ ? kBceInputsNumWithWeight : kBceInputsNumWithWeight - 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), expect_inputs_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kBceOutputsNum, kernel_name_);
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

void BinaryCrossEntropyCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  weight_defined_ = (input_num == kBceInputsNumWithWeight);
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

std::vector<KernelAttr> BinaryCrossEntropyCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    KernelAttr()
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeFloat16)
      .AddOutputAttr(kNumberTypeFloat16),
    KernelAttr()
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)};

  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BinaryCrossEntropy, BinaryCrossEntropyCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
