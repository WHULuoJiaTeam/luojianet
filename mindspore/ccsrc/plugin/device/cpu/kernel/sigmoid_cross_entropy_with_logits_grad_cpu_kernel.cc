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

#include "plugin/device/cpu/kernel/sigmoid_cross_entropy_with_logits_grad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSigmoidCrossEntropyWithLogitsGradInputsNum = 3;
constexpr size_t kSigmoidCrossEntropyWithLogitsGradOutputsNum = 1;
}  // namespace

void SigmoidCrossEntropyWithLogitsGradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  std::vector<size_t> x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  for (const uint64_t &d : x_shape) {
    tensor_size_ *= d;
  }
}

bool SigmoidCrossEntropyWithLogitsGradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                           const std::vector<kernel::AddressPtr> &,
                                                           const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSigmoidCrossEntropyWithLogitsGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSigmoidCrossEntropyWithLogitsGradOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32 || dtype_ == kNumberTypeFloat64) {
    LaunchKernel<float>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of input should be float16, float32, or float64, but got "
                      << TypeIdLabel(dtype_);
  }
  return true;
}

template <typename T>
void SigmoidCrossEntropyWithLogitsGradCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                                 const std::vector<AddressPtr> &outputs) {
  auto *logits_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto *labels_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto *dloss_addr = reinterpret_cast<T *>(inputs[2]->addr);
  auto *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  auto zero = static_cast<T>(0.0);
  auto one = static_cast<T>(1.0);
  for (uint64_t i = 0; i < tensor_size_; ++i) {
    if (logits_addr[i] >= zero) {
      output_addr[i] = (one / (one + static_cast<T>(exp(-logits_addr[i]))) - labels_addr[i]) * dloss_addr[i];
    } else {
      const T exp_val = static_cast<T>(exp(logits_addr[i]));
      output_addr[i] = (exp_val / (one + exp_val) - labels_addr[i]) * dloss_addr[i];
    }
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SigmoidCrossEntropyWithLogitsGrad,
                      SigmoidCrossEntropyWithLogitsGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
