/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FILL_V2_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FILL_V2_CPU_KERNEL_H_

#include <memory>
#include <vector>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class FillV2CpuKernelMod : public NativeCpuKernelMod {
 public:
  FillV2CpuKernelMod() = default;
  ~FillV2CpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  template <typename T>
  void LaunchKernel(AddressPtr *output, const AddressPtr &value);

  template <typename T>
  void CalculateDims(const AddressPtr &input, std::vector<int64_t> *dims);

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  TypeId output_dtype_{kTypeUnknown};
  TypeId input1_dtype_{kTypeUnknown};
  std::vector<size_t> output_shape_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FILL_V2_CPU_KERNEL_H_
