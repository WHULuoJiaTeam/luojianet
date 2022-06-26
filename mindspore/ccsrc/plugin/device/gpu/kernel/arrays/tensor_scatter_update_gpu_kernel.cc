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

#include "plugin/device/gpu/kernel/arrays/tensor_scatter_update_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(TensorScatterUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddOutputAttr(kNumberTypeFloat16),
                      TensorScatterUpdateFwdGpuKernelMod, half, int)
MS_REG_GPU_KERNEL_TWO(TensorScatterUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      TensorScatterUpdateFwdGpuKernelMod, float, int)
MS_REG_GPU_KERNEL_TWO(TensorScatterUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      TensorScatterUpdateFwdGpuKernelMod, double, int)
MS_REG_GPU_KERNEL_TWO(TensorScatterUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddOutputAttr(kNumberTypeInt8),
                      TensorScatterUpdateFwdGpuKernelMod, char, int)
MS_REG_GPU_KERNEL_TWO(TensorScatterUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddOutputAttr(kNumberTypeUInt8),
                      TensorScatterUpdateFwdGpuKernelMod, uchar, int)
MS_REG_GPU_KERNEL_TWO(TensorScatterUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      TensorScatterUpdateFwdGpuKernelMod, int, int)
MS_REG_GPU_KERNEL_TWO(TensorScatterUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      TensorScatterUpdateFwdGpuKernelMod, float, int64_t)
MS_REG_GPU_KERNEL_TWO(TensorScatterUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      TensorScatterUpdateFwdGpuKernelMod, double, int64_t)
MS_REG_GPU_KERNEL_TWO(TensorScatterUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeBool)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeBool)
                        .AddOutputAttr(kNumberTypeBool),
                      TensorScatterUpdateFwdGpuKernelMod, bool, int)
MS_REG_GPU_KERNEL_TWO(TensorScatterUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeBool)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeBool)
                        .AddOutputAttr(kNumberTypeBool),
                      TensorScatterUpdateFwdGpuKernelMod, bool, int64_t)
}  // namespace kernel
}  // namespace mindspore
