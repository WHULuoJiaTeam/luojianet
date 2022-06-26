/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/arrays/matrix_set_diag_gpu_kernel.h"
#include <algorithm>
#include <tuple>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "include/common/thread_pool.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(MatrixSetDiag,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      MatrixSetDiagGpuKernelMod, int)

MS_REG_GPU_KERNEL_ONE(MatrixSetDiag,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt64),
                      MatrixSetDiagGpuKernelMod, int64_t)

MS_REG_GPU_KERNEL_ONE(MatrixSetDiag,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      MatrixSetDiagGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(MatrixSetDiag,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat64),
                      MatrixSetDiagGpuKernelMod, double)
}  // namespace kernel
}  // namespace mindspore
