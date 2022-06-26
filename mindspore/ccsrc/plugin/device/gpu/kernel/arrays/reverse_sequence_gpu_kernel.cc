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

#include "plugin/device/gpu/kernel/arrays/reverse_sequence_gpu_kernel.h"

namespace mindspore {
namespace kernel {
// int8
MS_REG_GPU_KERNEL_TWO(
  ReverseSequence,
  KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
  ReverseSequenceFwdGpuKernelMod, int8_t, int)
MS_REG_GPU_KERNEL_TWO(
  ReverseSequence,
  KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
  ReverseSequenceFwdGpuKernelMod, int8_t, int64_t)
// int16
MS_REG_GPU_KERNEL_TWO(
  ReverseSequence,
  KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
  ReverseSequenceFwdGpuKernelMod, int16_t, int)
MS_REG_GPU_KERNEL_TWO(
  ReverseSequence,
  KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
  ReverseSequenceFwdGpuKernelMod, int16_t, int64_t)
// int32
MS_REG_GPU_KERNEL_TWO(
  ReverseSequence,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ReverseSequenceFwdGpuKernelMod, int, int)
MS_REG_GPU_KERNEL_TWO(
  ReverseSequence,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
  ReverseSequenceFwdGpuKernelMod, int, int64_t)
// int64
MS_REG_GPU_KERNEL_TWO(
  ReverseSequence,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
  ReverseSequenceFwdGpuKernelMod, int64_t, int)
MS_REG_GPU_KERNEL_TWO(
  ReverseSequence,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  ReverseSequenceFwdGpuKernelMod, int64_t, int64_t)
// float16
MS_REG_GPU_KERNEL_TWO(
  ReverseSequence,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  ReverseSequenceFwdGpuKernelMod, half, int)
MS_REG_GPU_KERNEL_TWO(
  ReverseSequence,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
  ReverseSequenceFwdGpuKernelMod, half, int64_t)
// float32
MS_REG_GPU_KERNEL_TWO(
  ReverseSequence,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  ReverseSequenceFwdGpuKernelMod, float, int)
MS_REG_GPU_KERNEL_TWO(
  ReverseSequence,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
  ReverseSequenceFwdGpuKernelMod, float, int64_t)
// float64
MS_REG_GPU_KERNEL_TWO(
  ReverseSequence,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  ReverseSequenceFwdGpuKernelMod, double, int)
MS_REG_GPU_KERNEL_TWO(
  ReverseSequence,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
  ReverseSequenceFwdGpuKernelMod, double, int64_t)
// bool
MS_REG_GPU_KERNEL_TWO(
  ReverseSequence,
  KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  ReverseSequenceFwdGpuKernelMod, bool, int)
MS_REG_GPU_KERNEL_TWO(
  ReverseSequence,
  KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  ReverseSequenceFwdGpuKernelMod, bool, int64_t)
}  // namespace kernel
}  // namespace mindspore
