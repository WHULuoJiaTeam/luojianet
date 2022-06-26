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

#include "plugin/device/gpu/kernel/rl/discounted_return_gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(DiscountedReturn,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeBool)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      DiscountedReturnGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(DiscountedReturn,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeBool)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddOutputAttr(kNumberTypeFloat16),
                      DiscountedReturnGpuKernelMod, half)
}  // namespace kernel
}  // namespace mindspore
