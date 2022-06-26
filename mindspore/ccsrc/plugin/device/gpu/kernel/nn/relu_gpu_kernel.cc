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

#include "plugin/device/gpu/kernel/nn/relu_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(ReLU, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      ReLUFwdGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(ReLU, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      ReLUFwdGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(ReLU, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      ReLUFwdGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(ReLU, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      ReLUFwdGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(ReLU, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      ReLUFwdGpuKernelMod, int32_t)
MS_REG_GPU_KERNEL_ONE(ReLU, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      ReLUFwdGpuKernelMod, int16_t)
MS_REG_GPU_KERNEL_ONE(ReLU, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      ReLUFwdGpuKernelMod, int8_t)
MS_REG_GPU_KERNEL_ONE(ReLU, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      ReLUFwdGpuKernelMod, uint8_t)
}  // namespace kernel
}  // namespace mindspore
