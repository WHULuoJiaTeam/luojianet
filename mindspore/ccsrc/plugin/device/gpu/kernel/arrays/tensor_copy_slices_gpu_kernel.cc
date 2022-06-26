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

#include "plugin/device/gpu/kernel/arrays/tensor_copy_slices_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(
  TensorCopySlices,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  TensorCopySlicesGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(
  TensorCopySlices,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  TensorCopySlicesGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  TensorCopySlices,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  TensorCopySlicesGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  TensorCopySlices,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  TensorCopySlicesGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(
  TensorCopySlices,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  TensorCopySlicesGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(
  TensorCopySlices,
  KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
  TensorCopySlicesGpuKernelMod, char)
MS_REG_GPU_KERNEL_ONE(
  TensorCopySlices,
  KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
  TensorCopySlicesGpuKernelMod, uchar)
}  // namespace kernel
}  // namespace mindspore
