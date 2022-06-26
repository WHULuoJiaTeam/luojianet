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

#include "plugin/device/gpu/kernel/arrays/spacetobatch_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(SpaceToBatch, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      SpaceToBatchGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(SpaceToBatch, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      SpaceToBatchGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(SpaceToBatch, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      SpaceToBatchGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(SpaceToBatch, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      SpaceToBatchGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(SpaceToBatch, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      SpaceToBatchGpuKernelMod, int16_t)
MS_REG_GPU_KERNEL_ONE(SpaceToBatch, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      SpaceToBatchGpuKernelMod, int8_t)
MS_REG_GPU_KERNEL_ONE(SpaceToBatch, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      SpaceToBatchGpuKernelMod, uint8_t)
MS_REG_GPU_KERNEL_ONE(SpaceToBatch, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                      SpaceToBatchGpuKernelMod, uint16_t)
MS_REG_GPU_KERNEL_ONE(SpaceToBatch, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      SpaceToBatchGpuKernelMod, uint32_t)
MS_REG_GPU_KERNEL_ONE(SpaceToBatch, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                      SpaceToBatchGpuKernelMod, uint64_t)
}  // namespace kernel
}  // namespace mindspore
