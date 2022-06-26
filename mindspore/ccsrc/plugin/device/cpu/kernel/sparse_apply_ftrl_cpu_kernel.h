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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_APPLY_FTRL_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_APPLY_FTRL_CPU_KERNEL_H_

#include <vector>
#include "plugin/device/cpu/kernel/sparse_optimizer_cpu_kernel.h"

namespace mindspore {
namespace kernel {
class BACKEND_EXPORT SparseApplyFtrlCpuKernelMod : public SparseOptimizerCpuKernelMod {
 public:
  SparseApplyFtrlCpuKernelMod() = default;
  ~SparseApplyFtrlCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  float lr_{0.0};
  float l1_{0.0};
  float l2_{0.0};
  float lr_power_{0.0};

  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {KernelAttr()
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeInt32)
                                                     .AddOutputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32),
                                                   KernelAttr()
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddOutputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32)};
    return support_list;
  }

 private:
  void InitInputOutputSize(const CNodePtr &kernel_node) override;

  template <typename T>
  void InitWorkspaceSize();

  template <typename T>
  void LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                    const std::vector<kernel::AddressPtr> &workspace) const;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_APPLY_FTRL_CPU_KERNEL_H_
