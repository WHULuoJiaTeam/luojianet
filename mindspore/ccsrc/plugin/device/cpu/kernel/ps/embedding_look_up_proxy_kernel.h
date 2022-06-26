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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EMBEDDING_LOOK_UP_PROXY_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EMBEDDING_LOOK_UP_PROXY_KERNEL_H_

#include "plugin/device/cpu/kernel/embedding_look_up_cpu_kernel.h"
#include <vector>
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace ps {
class EmbeddingLookUpProxyKernel : public EmbeddingLookUpCpuKernelMod {
 public:
  EmbeddingLookUpProxyKernel() = default;
  ~EmbeddingLookUpProxyKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32)};
    return support_list;
  }

 private:
  size_t key_{0};
  size_t input_dims_{1};
};
}  // namespace ps
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EMBEDDING_LOOK_UP_PROXY_KERNEL_H_
