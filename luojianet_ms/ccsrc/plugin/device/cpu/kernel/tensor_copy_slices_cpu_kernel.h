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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TENSOR_COPY_SLICES_CPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TENSOR_COPY_SLICES_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "nnacl/fp32/strided_slice_fp32.h"

namespace luojianet_ms {
namespace kernel {
class TensorCopySlicesCpuKernelMod : public NativeCpuKernelMod {
 public:
  TensorCopySlicesCpuKernelMod() = default;
  ~TensorCopySlicesCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  TypeId data_type_;
  size_t offset_{0};
  size_t copy_size_{0};
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> update_shape_;
  std::vector<int64_t> output_shape_;
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TENSOR_COPY_SLICES_CPU_KERNEL_H_
