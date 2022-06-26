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
#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FUSED_ADA_FACTOR_CPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FUSED_ADA_FACTOR_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <map>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace luojianet_ms {
namespace kernel {
constexpr auto kFusedAdaFactor = "FusedAdaFactor";
constexpr auto kFusedAdaFactorWithGlobalNorm = "FusedAdaFactorWithGlobalNorm";
constexpr auto kUnknown = "Unknown";
class FusedAdaFactorCpuKernelMod : public NativeCpuKernelMod {
 public:
  FusedAdaFactorCpuKernelMod() = default;
  explicit FusedAdaFactorCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~FusedAdaFactorCpuKernelMod() override = default;
  void InitKernel(const CNodePtr &kernel_node) override;
  void InitInputOutputSize(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void CheckInputAddresses(const std::vector<AddressPtr> &inputs) const;
  void CheckWorkspaceAddresses(const std::vector<AddressPtr> &workspaces) const;

  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
                    const std::vector<AddressPtr> &outputs);

  template <typename T>
  float CalcRMS(T *input, size_t elem_num);

  template <typename T>
  void FactorUpdate(float *update, const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces);

  bool enable_scale_parameter_{false};
  bool enable_first_moment_{false};
  bool enable_weight_decay_{false};
  bool need_factor_{false};
  size_t elem_num_{0};
  size_t last_row_dim_size_{1};
  size_t last_col_dim_size_{1};
  TypeId param_dtype_{kTypeUnknown};
  float global_norm_reciprocal_{1.0f};
  std::string kernel_type_{kUnknown};
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FUSED_ADA_FACTOR_CPU_KERNEL_H_
