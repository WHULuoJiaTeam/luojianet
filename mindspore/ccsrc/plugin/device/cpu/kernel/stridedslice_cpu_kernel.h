/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_STRIDESLICE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_STRIDESLICE_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "nnacl/fp32/strided_slice_fp32.h"

namespace mindspore {
namespace kernel {
class StridedSliceCpuKernelMod : public NativeCpuKernelMod {
 public:
  StridedSliceCpuKernelMod() = default;
  ~StridedSliceCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
      KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
      KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
      KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
      KernelAttr()
        .AddInputAttr(kNumberTypeBool)
        .AddInputAttr(kNumberTypeInt64)
        .AddInputAttr(kNumberTypeInt64)
        .AddInputAttr(kNumberTypeInt64)
        .AddOutputAttr(kNumberTypeBool),
      KernelAttr()
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt64)
        .AddInputAttr(kNumberTypeInt64)
        .AddInputAttr(kNumberTypeInt64)
        .AddOutputAttr(kNumberTypeInt32),
      KernelAttr()
        .AddInputAttr(kNumberTypeFloat32)
        .AddInputAttr(kNumberTypeInt64)
        .AddInputAttr(kNumberTypeInt64)
        .AddInputAttr(kNumberTypeInt64)
        .AddOutputAttr(kNumberTypeFloat32),
      KernelAttr()
        .AddInputAttr(kNumberTypeFloat64)
        .AddInputAttr(kNumberTypeInt64)
        .AddInputAttr(kNumberTypeInt64)
        .AddInputAttr(kNumberTypeInt64)
        .AddOutputAttr(kNumberTypeFloat64)};
    return support_list;
  }

 private:
  enum ParallelStrategy { kOnSplitAxis, kOnOuter };
  void InitSliceParam(const CNodePtr &kernel_node, std::vector<int64_t> *begin, std::vector<int64_t> *end,
                      std::vector<int64_t> *stride);
  bool MatchParallelPattern();
  void InitParallelParam();
  void ParallelRun(const uint8_t *input_addr, uint8_t *output_addr, int thread_num);
  common::Status RunTaskOnOuter(const uint8_t *input_addr, uint8_t *output_addr, int start_pos);
  common::Status RunTaskOnSplitAxis(const uint8_t *input_addr, uint8_t *output_addr, int start_pos);
  void ParseMasks(const CNodePtr &kernel_node);

  TypeId dtype_;
  int data_size_{4};
  int split_axis_{-1};
  int inner_{1};
  int outer_{1};
  int cal_num_per_thread_{1};
  bool parallel_{false};
  ParallelStrategy parallel_strategy_{kOnSplitAxis};
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  StridedSliceParameter slice_param_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_STRIDESLICE_CPU_KERNEL_H_
