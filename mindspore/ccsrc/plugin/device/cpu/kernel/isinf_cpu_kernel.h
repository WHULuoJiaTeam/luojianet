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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_IsInf_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_IsInf_CPU_KERNEL_H_

#include <vector>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class IsInfCpuKernelMod : public NativeCpuKernelMod {
 public:
  IsInfCpuKernelMod() = default;
  ~IsInfCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernelNode) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
      KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
      KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool)};
    return support_list;
  }

 private:
  template <typename T>
  void LaunchKernelFloat(const std::vector<AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

  void LaunchKernelFloat16(const std::vector<AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

  std::map<TypeId, size_t> dtype_map_ = {
    {kNumberTypeFloat16, sizeof(float16)}, {kNumberTypeFloat32, sizeof(float)}, {kNumberTypeFloat64, sizeof(double)}};

  TypeId input_dtype_{kTypeUnknown};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_IsInf_CPU_KERNEL_H_
