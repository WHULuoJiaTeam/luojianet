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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SPARSE_APPLY_FTRL_PS_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SPARSE_APPLY_FTRL_PS_KERNEL_H_

#include <vector>
#include <memory>
#include "plugin/device/cpu/kernel/ps/pserver_kernel.h"
#include "plugin/device/cpu/kernel/sparse_apply_ftrl_cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace ps {
using mindspore::kernel::SparseApplyFtrlCpuKernelMod;
class SparseApplyFtrlPSKernelMod : public SparseApplyFtrlCpuKernelMod, public PServerKernel {
 public:
  SparseApplyFtrlPSKernelMod(size_t rank_id, size_t pserver_num, size_t worker_num)
      : PServerKernel(rank_id, pserver_num, worker_num), init_accum_(0.1) {}
  ~SparseApplyFtrlPSKernelMod() override = default;

  void InitKernel(const CNodePtr &cnode,
                  const std::shared_ptr<std::vector<std::shared_ptr<std::vector<size_t>>>> &) override;
  void ReInit(const std::vector<std::vector<size_t>> &) override;

  bool Execute(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override;

  const std::vector<size_t> &input_sizes() const override;
  const std::vector<size_t> &output_sizes() const override;
  const std::vector<size_t> &workspace_sizes() const override;
  const float init_accum() const { return init_accum_; }

 protected:
  void ReInit(const std::vector<AddressPtr> &) override;
  float init_accum_{0.1};
  size_t var_index_{0};
  size_t accum_index_{1};
  size_t linear_index_{2};
  size_t grad_index_{3};
  size_t indices_index_{4};
};
}  // namespace ps
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SPARSE_APPLY_FTRL_PS_KERNEL_H_
