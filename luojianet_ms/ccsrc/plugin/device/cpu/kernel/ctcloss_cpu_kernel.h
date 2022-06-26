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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CTCLOSS_CPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CTCLOSS_CPU_KERNEL_H_

#include <memory>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <limits>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace luojianet_ms {
namespace kernel {
class CTCLossCpuKernelMod : public NativeCpuKernelMod {
 public:
  CTCLossCpuKernelMod() = default;
  ~CTCLossCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void GenLabelWithBlank(const uint32_t *seq_len, const std::vector<std::vector<uint32_t>> &batch_label,
                         std::vector<std::vector<uint32_t>> *label_with_blank) const;

  template <typename T>
  void CalculateFwdVar(const std::vector<uint32_t> &label_with_blank, const std::vector<std::vector<T>> &y,
                       std::vector<std::vector<T>> *log_alpha_b) const;
  template <typename T>
  void CalculateBwdVar(const std::vector<uint32_t> &label_with_blank, const std::vector<std::vector<T>> &y,
                       std::vector<std::vector<T>> *log_beta_b) const;
  template <typename T>
  void CalculateGrad(const std::vector<uint32_t> &label_with_blank, const std::vector<std::vector<T>> &y,
                     const std::vector<std::vector<T>> &log_alpha_b, const std::vector<std::vector<T>> &log_beta_b,
                     const T log_pzx, std::vector<std::vector<T>> *dy) const;

  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) const;

  std::vector<size_t> probs_shape_;
  std::vector<size_t> indices_dims_;
  std::vector<size_t> labels_dims_;
  size_t num_class_{0};
  size_t max_time_{0};
  size_t batch_size_{0};
  uint32_t blank_index_{0};
  TypeId dtype_{kTypeUnknown};
  bool preprocess_collapse_repeated_{false};
  bool ctc_merge_repeated_{false};
  bool ignore_longer_outputs_than_inputs_{false};
};
}  // namespace kernel
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CTCLOSS_CPU_KERNEL_H_
