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

#include "plugin/device/cpu/kernel/expm1_cpu_kernel.h"
#include <cmath>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kExpm1InputsNum = 1;
constexpr size_t kExpm1OutputsNum = 1;
}  // namespace

void Expm1CpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  if (input_dtype_ != kNumberTypeFloat16 && input_dtype_ != kNumberTypeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of input should be Float16 or Float32, but got: " << input_dtype_;
  }
}

bool Expm1CpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                               const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kExpm1InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kExpm1OutputsNum, kernel_name_);
  if (input_dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of input should be Float16 or Float32, but got: " << TypeIdLabel(input_dtype_);
  }
  return true;
}

template <typename T>
void Expm1CpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);
  size_t elem_num = inputs[0]->size / sizeof(T);
  for (size_t i = 0; i < elem_num; i++) {
    output[i] = exp(input[i]) - T(1);
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Expm1, Expm1CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
