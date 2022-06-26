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

#include "plugin/device/gpu/kernel/arrays/gatherv2_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(
  Gather,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  GatherV2FwdGpuKernelMod, double, int)
MS_REG_GPU_KERNEL_TWO(
  Gather,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
  GatherV2FwdGpuKernelMod, double, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Gather,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  GatherV2FwdGpuKernelMod, float, int)
MS_REG_GPU_KERNEL_TWO(
  Gather,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
  GatherV2FwdGpuKernelMod, float, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Gather,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  GatherV2FwdGpuKernelMod, half, int)
MS_REG_GPU_KERNEL_TWO(
  Gather,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
  GatherV2FwdGpuKernelMod, half, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Gather, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  GatherV2FwdGpuKernelMod, int, int)
MS_REG_GPU_KERNEL_TWO(
  Gather, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
  GatherV2FwdGpuKernelMod, int, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Gather, KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
  GatherV2FwdGpuKernelMod, int16_t, int)
MS_REG_GPU_KERNEL_TWO(
  Gather, KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
  GatherV2FwdGpuKernelMod, int16_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Gather, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
  GatherV2FwdGpuKernelMod, int8_t, int)
MS_REG_GPU_KERNEL_TWO(
  Gather, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
  GatherV2FwdGpuKernelMod, int8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Gather, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
  GatherV2FwdGpuKernelMod, uint, int)
MS_REG_GPU_KERNEL_TWO(
  Gather, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
  GatherV2FwdGpuKernelMod, uint, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Gather, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
  GatherV2FwdGpuKernelMod, uint8_t, int)
MS_REG_GPU_KERNEL_TWO(
  Gather, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
  GatherV2FwdGpuKernelMod, uint8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Gather, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  GatherV2FwdGpuKernelMod, bool, int)
MS_REG_GPU_KERNEL_TWO(
  Gather, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  GatherV2FwdGpuKernelMod, bool, int64_t)
// dynamic shape
MS_REG_GPU_KERNEL_TWO(Gather,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      GatherV2FwdGpuKernelMod, double, int)
MS_REG_GPU_KERNEL_TWO(Gather,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      GatherV2FwdGpuKernelMod, double, int64_t)
MS_REG_GPU_KERNEL_TWO(Gather,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat32),
                      GatherV2FwdGpuKernelMod, float, int)
MS_REG_GPU_KERNEL_TWO(Gather,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat32),
                      GatherV2FwdGpuKernelMod, float, int64_t)
MS_REG_GPU_KERNEL_TWO(Gather,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat16),
                      GatherV2FwdGpuKernelMod, half, int)
MS_REG_GPU_KERNEL_TWO(Gather,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat16),
                      GatherV2FwdGpuKernelMod, half, int64_t)
MS_REG_GPU_KERNEL_TWO(Gather,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeBool)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeBool),
                      GatherV2FwdGpuKernelMod, bool, int)
MS_REG_GPU_KERNEL_TWO(Gather,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeBool)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeBool),
                      GatherV2FwdGpuKernelMod, bool, int64_t)
// dynamic shape ends
MS_REG_GPU_KERNEL_TWO(
  SparseGatherV2,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  GatherV2FwdGpuKernelMod, float, int)
MS_REG_GPU_KERNEL_TWO(
  SparseGatherV2,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  GatherV2FwdGpuKernelMod, half, int)
MS_REG_GPU_KERNEL_TWO(SparseGatherV2,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat32),
                      GatherV2FwdGpuKernelMod, float, int)
MS_REG_GPU_KERNEL_TWO(SparseGatherV2,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat16),
                      GatherV2FwdGpuKernelMod, half, int)
}  // namespace kernel
}  // namespace mindspore
