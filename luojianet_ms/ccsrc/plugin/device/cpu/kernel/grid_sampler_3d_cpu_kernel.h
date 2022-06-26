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
#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GRID_SAMPLER_3D_CPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GRID_SAMPLER_3D_CPU_KERNEL_H_
#include <algorithm>
#include <vector>
#include <memory>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace luojianet_ms {
namespace kernel {
class GridSampler3DCpuKernelMod : public NativeCpuKernelMod {
 public:
  GridSampler3DCpuKernelMod() = default;
  ~GridSampler3DCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

 protected:
  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
      KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};
    return support_list;
  }

 private:
  std::vector<size_t> x_shape_;
  std::vector<size_t> grid_shape_;
  std::vector<size_t> output_shape_;
  std::vector<size_t> x_stride_;
  std::vector<size_t> grid_stride_;
  std::vector<size_t> output_stride_;
  std::string interpolation_mode;
  std::string padding_mode;
  bool align_corners;
  size_t output_number_;
  TypeId dtype_{kTypeUnknown};
  template <typename T>
  void ComputeTask(T *x_data_addr, T *grid_data_addr, T *output_data_addr, const size_t &seq);

  template <typename T>
  T grid_sampler_compute_source_index(T coord, int64_t size, const std::string &padding_mode, bool align_corners);

  template <typename T>
  T reflect_coordinates(T coord, int64_t twice_low, int64_t twice_high);

  bool within_bounds_3d(int64_t d, int64_t h, int64_t w, size_t D, size_t H, size_t W);
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GRID_SAMPLER_3D_CPU_KERNEL_H_
