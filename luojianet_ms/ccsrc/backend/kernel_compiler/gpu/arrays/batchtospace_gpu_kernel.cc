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

#include "backend/kernel_compiler/gpu/arrays/batchtospace_gpu_kernel.h"

namespace luojianet_ms {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(BatchToSpace, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      BatchToSpaceGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(BatchToSpace, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      BatchToSpaceGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(BatchToSpace, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      BatchToSpaceGpuKernel, int)
MS_REG_GPU_KERNEL_ONE(BatchToSpace, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      BatchToSpaceGpuKernel, int64_t)
MS_REG_GPU_KERNEL_ONE(BatchToSpace, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      BatchToSpaceGpuKernel, int16_t)
MS_REG_GPU_KERNEL_ONE(BatchToSpace, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      BatchToSpaceGpuKernel, int8_t)
MS_REG_GPU_KERNEL_ONE(BatchToSpace, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      BatchToSpaceGpuKernel, uint8_t)
MS_REG_GPU_KERNEL_ONE(BatchToSpace, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                      BatchToSpaceGpuKernel, uint16_t)
MS_REG_GPU_KERNEL_ONE(BatchToSpace, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      BatchToSpaceGpuKernel, uint32_t)
MS_REG_GPU_KERNEL_ONE(BatchToSpace, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                      BatchToSpaceGpuKernel, uint64_t)
}  // namespace kernel
}  // namespace luojianet_ms
