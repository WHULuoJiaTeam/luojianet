/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_EIGEN_MATMUL_DOULE_CPU_KERNEL_FUNC_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_EIGEN_MATMUL_DOULE_CPU_KERNEL_FUNC_H_

#include <string>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace kernel {
class MatmulDoubleCpuKernelFunc : public CpuKernelFunc {
 public:
  MatmulDoubleCpuKernelFunc() = default;
  ~MatmulDoubleCpuKernelFunc() override = default;

  void InitFunc(const CNodePtr &kernel_node) override;
  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override;

 private:
  size_t a_row_{0};
  size_t b_row_{0};
  size_t out_row_{0};
  size_t a_col_{0};
  size_t b_col_{0};
  size_t out_col_{0};
  bool trans_a_{false};
  bool trans_b_{false};
  std::string kernel_name_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_EIGEN_MATMUL_DOULE_CPU_KERNEL_FUNC_H_
