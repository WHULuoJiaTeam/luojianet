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
#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_HSV_TO_RGB_CPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_HSV_TO_RGB_CPU_KERNEL_H_
#include <vector>
#include <memory>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace luojianet_ms {
namespace kernel {
class HSVToRGBCpuKernelMod : public NativeCpuKernelMod {
 public:
  HSVToRGBCpuKernelMod() = default;
  ~HSVToRGBCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
      KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
      KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};
    return support_list;
  }

 private:
  TypeId input_dtype;
  template <typename T1>
  void ConvertOnePixel(T1 h, T1 s, T1 v, T1 *r, T1 *g, T1 *b);
  template <typename T1>
  void ComputeFloat(void *input, void *output, int64_t pixel_num);
  void ComputeHalf(void *input, void *output, int64_t pixel_num);
  std::vector<size_t> shape;
  const size_t kInputNum = 1;
  const size_t kOutputNum = 1;
};
}  // namespace kernel
}  // namespace luojianet_ms
#endif
