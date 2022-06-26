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

#include "plugin/device/gpu/kernel/arrays/gather_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(
  GatherD,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  GatherFwdGpuKernelMod, double, int)
MS_REG_GPU_KERNEL_TWO(
  GatherD,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
  GatherFwdGpuKernelMod, double, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherD,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  GatherFwdGpuKernelMod, float, int)
MS_REG_GPU_KERNEL_TWO(
  GatherD,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
  GatherFwdGpuKernelMod, float, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherD,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  GatherFwdGpuKernelMod, half, int)
MS_REG_GPU_KERNEL_TWO(
  GatherD,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
  GatherFwdGpuKernelMod, half, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherD, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  GatherFwdGpuKernelMod, int, int)
MS_REG_GPU_KERNEL_TWO(
  GatherD, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
  GatherFwdGpuKernelMod, int, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherD, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
  GatherFwdGpuKernelMod, int8_t, int)
MS_REG_GPU_KERNEL_TWO(
  GatherD, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
  GatherFwdGpuKernelMod, int8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherD, KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
  GatherFwdGpuKernelMod, int16_t, int)
MS_REG_GPU_KERNEL_TWO(
  GatherD, KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
  GatherFwdGpuKernelMod, int16_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherD, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
  GatherFwdGpuKernelMod, int64_t, int)
MS_REG_GPU_KERNEL_TWO(
  GatherD, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  GatherFwdGpuKernelMod, int64_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherD, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
  GatherFwdGpuKernelMod, uint, int)
MS_REG_GPU_KERNEL_TWO(
  GatherD, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
  GatherFwdGpuKernelMod, uint, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherD, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
  GatherFwdGpuKernelMod, uchar, int)
MS_REG_GPU_KERNEL_TWO(
  GatherD, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
  GatherFwdGpuKernelMod, uchar, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherD, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  GatherFwdGpuKernelMod, bool, int)
MS_REG_GPU_KERNEL_TWO(
  GatherD, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  GatherFwdGpuKernelMod, bool, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherD, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
  GatherFwdGpuKernelMod, uint32_t, int)
MS_REG_GPU_KERNEL_TWO(
  GatherD, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
  GatherFwdGpuKernelMod, uint32_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherD, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
  GatherFwdGpuKernelMod, uint64_t, int)
MS_REG_GPU_KERNEL_TWO(
  GatherD, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
  GatherFwdGpuKernelMod, uint64_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherD, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
  GatherFwdGpuKernelMod, uint16_t, int)
MS_REG_GPU_KERNEL_TWO(
  GatherD, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
  GatherFwdGpuKernelMod, uint16_t, int64_t)
}  // namespace kernel
}  // namespace mindspore
