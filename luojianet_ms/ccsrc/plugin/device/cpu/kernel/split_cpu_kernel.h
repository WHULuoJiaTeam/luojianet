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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPLIT_CPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPLIT_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <thread>
#include <tuple>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/nnacl/base/split_base.h"

namespace luojianet_ms {
namespace kernel {
class SplitCpuKernelMod : public NativeCpuKernelMod {
 public:
  SplitCpuKernelMod() = default;
  ~SplitCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void CheckParam(const CNodePtr &kernel_node);
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);
  template <typename T>
  void InitIOSize(const CNodePtr &kernel_node);

  using SplitFunc = std::function<bool(SplitCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                       const std::vector<AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  using InitIOFunc = std::function<void(SplitCpuKernelMod *, const CNodePtr &)>;
  static std::vector<std::tuple<KernelAttr, SplitFunc, InitIOFunc>> func_list_;
  SplitFunc kernel_func_;
  InitIOFunc init_io_func_;

  template <typename T>
  void LaunchSplit(T *input, T **output, size_t size);

  void InitInputOutputSize(const CNodePtr &kernel_node) override { init_io_func_(this, kernel_node); }

  int64_t axis_{0};
  size_t output_num_{1};
  std::vector<int> input_shape_;
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPLIT_CPU_KERNEL_H_
