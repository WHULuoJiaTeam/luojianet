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

#include "plugin/device/gpu/kernel/arrays/tensor_scatter_min_gpu_kernel.h"

namespace mindspore {
namespace kernel {
// for int32 index
MS_REG_GPU_KERNEL_TWO(TensorScatterMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddOutputAttr(kNumberTypeFloat16),
                      TensorScatterMinGpuKernelMod, half, int32_t)

MS_REG_GPU_KERNEL_TWO(TensorScatterMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      TensorScatterMinGpuKernelMod, float, int32_t)

MS_REG_GPU_KERNEL_TWO(TensorScatterMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddOutputAttr(kNumberTypeInt8),
                      TensorScatterMinGpuKernelMod, char, int32_t)

MS_REG_GPU_KERNEL_TWO(TensorScatterMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddOutputAttr(kNumberTypeUInt8),
                      TensorScatterMinGpuKernelMod, uchar, int32_t)

MS_REG_GPU_KERNEL_TWO(TensorScatterMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      TensorScatterMinGpuKernelMod, int32_t, int32_t)

// for int64 index
MS_REG_GPU_KERNEL_TWO(TensorScatterMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddOutputAttr(kNumberTypeFloat16),
                      TensorScatterMinGpuKernelMod, half, int64_t)

MS_REG_GPU_KERNEL_TWO(TensorScatterMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      TensorScatterMinGpuKernelMod, float, int64_t)

MS_REG_GPU_KERNEL_TWO(TensorScatterMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddOutputAttr(kNumberTypeInt8),
                      TensorScatterMinGpuKernelMod, char, int64_t)

MS_REG_GPU_KERNEL_TWO(TensorScatterMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddOutputAttr(kNumberTypeUInt8),
                      TensorScatterMinGpuKernelMod, uchar, int64_t)

MS_REG_GPU_KERNEL_TWO(TensorScatterMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      TensorScatterMinGpuKernelMod, int32_t, int64_t)
}  // namespace kernel
}  // namespace mindspore
