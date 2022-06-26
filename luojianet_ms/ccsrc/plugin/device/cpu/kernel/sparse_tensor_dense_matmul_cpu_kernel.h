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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_TENSOR_DENSE_MATMUL_CPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_TENSOR_DENSE_MATMUL_CPU_KERNEL_H_

#include <vector>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace luojianet_ms {
namespace kernel {
class SparseTensorDenseMatmulCpuKernelMod : public NativeCpuKernelMod {
 public:
  SparseTensorDenseMatmulCpuKernelMod() = default;
  ~SparseTensorDenseMatmulCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  using SparseTensorDenseMatmulFunc =
    std::function<bool(SparseTensorDenseMatmulCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, SparseTensorDenseMatmulFunc>> func_list_;
  SparseTensorDenseMatmulFunc kernel_func_;

  std::vector<size_t> output_shape_;
  std::vector<size_t> b_shape_;
  size_t output_size_{0};
  size_t values_size_{0};
  bool adj_st_{false};
  bool adj_dt_{false};
  enum input_list_ { INDICES, VALUES, SPARSE_SHAPE, DENSE };
};
}  // namespace kernel
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RMSPROP_CPU_KERNEL_H_
