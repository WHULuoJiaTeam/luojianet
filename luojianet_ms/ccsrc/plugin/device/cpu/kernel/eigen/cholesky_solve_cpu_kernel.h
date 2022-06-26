/**
 * Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022 Huawei Technologies Co., Ltd
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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CHOLESKY_CPU_SOLVE_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CHOLESKY_CPU_SOLVE_KERNEL_H_
#include <vector>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace luojianet_ms {
namespace kernel {
class CholeskySolveCpuKernelMod : public NativeCpuKernelMod {
 public:
  CholeskySolveCpuKernelMod() = default;
  ~CholeskySolveCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void InitLeftMatrixInfo(const std::vector<size_t> &shape, const bool is_rank_equal, size_t *row, size_t *col);
  void InitRightMatrixInfo(const std::vector<size_t> &shape, size_t *row, size_t *col);

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);
  using CholeskySolveFunc =
    std::function<bool(CholeskySolveCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, CholeskySolveFunc>> func_list_;
  CholeskySolveFunc kernel_func_;

  size_t outer_batch_{1};
  size_t input_a_row_{1};
  size_t input_a_col_{1};
  size_t input_b_row_{1};
  size_t input_b_col_{1};
  size_t output_row_{1};
  size_t output_col_{1};
  TypeId dtype_{kNumberTypeFloat32};
  bool lower_{false};
};
}  // namespace kernel
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CHOLESKY_CPU_SOLVE_KERNEL_H_
