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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GATHER_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GATHER_CPU_KERNEL_H_

#include <utility>
#include <vector>
#include <memory>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/nnacl/base/gather_base.h"

namespace mindspore {
namespace kernel {
class GatherV2CpuKernelMod : public NativeCpuKernelMod {
 public:
  GatherV2CpuKernelMod() = default;
  ~GatherV2CpuKernelMod() override = default;

  void CheckParam(const CNodePtr &kernel_node);

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  using GatherV2Func = std::function<bool(GatherV2CpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                          const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, GatherV2Func>> func_list_;
  GatherV2Func kernel_func_;

  std::vector<size_t> input_shape_;
  std::vector<size_t> indices_shape_;
  std::vector<size_t> output_shape_;
  int64_t axis_{0};
  bool is_dynamic_shape_{false};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GATHER_CPU_KERNEL_H_
