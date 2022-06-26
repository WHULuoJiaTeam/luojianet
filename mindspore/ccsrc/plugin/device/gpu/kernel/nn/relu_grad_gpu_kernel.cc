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

#include "plugin/device/gpu/kernel/nn/relu_grad_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(
  ReluGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  ReluGradFwdGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(
  ReluGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ReluGradFwdGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  ReluGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  ReluGradFwdGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  ReluGrad, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  ReluGradFwdGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(
  ReluGrad, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ReluGradFwdGpuKernelMod, int32_t)
MS_REG_GPU_KERNEL_ONE(
  ReluGrad, KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
  ReluGradFwdGpuKernelMod, int16_t)
MS_REG_GPU_KERNEL_ONE(
  ReluGrad, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
  ReluGradFwdGpuKernelMod, int8_t)
MS_REG_GPU_KERNEL_ONE(
  ReluGrad, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
  ReluGradFwdGpuKernelMod, uint8_t)
}  // namespace kernel
}  // namespace mindspore
