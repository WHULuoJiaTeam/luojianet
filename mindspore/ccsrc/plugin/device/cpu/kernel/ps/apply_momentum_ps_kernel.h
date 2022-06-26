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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_APPLY_MOMENTUM_PS_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_APPLY_MOMENTUM_PS_KERNEL_H_

#include <vector>
#include <memory>
#include "plugin/device/cpu/kernel/ps/pserver_kernel.h"
#include "plugin/device/cpu/kernel/apply_momentum_cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace ps {
class ApplyMomentumPSKernelMod : public ApplyMomentumCpuKernelMod, public PServerKernel {
 public:
  ApplyMomentumPSKernelMod(size_t rank_id, size_t pserver_num, size_t worker_num)
      : PServerKernel(rank_id, pserver_num, worker_num) {}
  ~ApplyMomentumPSKernelMod() override = default;

  bool Execute(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override;

  const std::vector<size_t> &input_sizes() const override;
  const std::vector<size_t> &output_sizes() const override;
  const std::vector<size_t> &workspace_sizes() const override;
};
}  // namespace ps
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_APPLY_MOMENTUM_PS_KERNEL_H_
