/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/math/unary_op_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(Exp, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Exp, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Expm1, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Expm1, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Log, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      UnaryOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(Log, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Log, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Log1p, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Log1p, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Erf, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Erf, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Erfc, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Erfc, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Neg, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      UnaryOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(Neg, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Neg, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Neg, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      UnaryOpGpuKernelMod, char)
MS_REG_GPU_KERNEL_ONE(Neg, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      UnaryOpGpuKernelMod, uchar)
MS_REG_GPU_KERNEL_ONE(Neg, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      UnaryOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(Reciprocal, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Reciprocal, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Square, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Square, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Sqrt, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      UnaryOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(Sqrt, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Sqrt, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Rsqrt, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Rsqrt, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Sin, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      UnaryOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(Sin, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Sin, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Asin, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Asin, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Asinh, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Asinh, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Cos, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      UnaryOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(Cos, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Cos, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(ACos, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(ACos, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Acosh, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Acosh, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Atan, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Atan, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Abs, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      UnaryOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(Abs, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Abs, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Abs, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      UnaryOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(Floor, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Floor, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Rint, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      UnaryOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(Rint, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Rint, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Round, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      UnaryOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(Round, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      UnaryOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(Round, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Round, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Sign, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      UnaryOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(Sign, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      UnaryOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(Sign, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Sign, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpGpuKernelMod, half)
}  // namespace kernel
}  // namespace mindspore
