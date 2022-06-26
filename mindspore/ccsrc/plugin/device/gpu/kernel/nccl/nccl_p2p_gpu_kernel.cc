/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/nccl/nccl_p2p_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(
  AllToAllv, KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  NcclP2PGpuKernel, float, float)
MS_REG_GPU_KERNEL_TWO(
  AllToAllv, KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  NcclP2PGpuKernel, half, half)
MS_REG_GPU_KERNEL_TWO(AllToAllv,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      NcclP2PGpuKernel, int, int)
MS_REG_GPU_KERNEL_TWO(
  AllToAllv, KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat16),
  NcclP2PGpuKernel, float, half)
MS_REG_GPU_KERNEL_TWO(
  AllToAllv, KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat32),
  NcclP2PGpuKernel, half, float)

MS_REG_GPU_KERNEL_TWO(
  NeighborExchange,
  KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  NcclP2PGpuKernel, float, float)
MS_REG_GPU_KERNEL_TWO(
  NeighborExchange,
  KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  NcclP2PGpuKernel, half, half)
MS_REG_GPU_KERNEL_TWO(NeighborExchange,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      NcclP2PGpuKernel, int, int)
MS_REG_GPU_KERNEL_TWO(
  NeighborExchange,
  KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat16),
  NcclP2PGpuKernel, float, half)
MS_REG_GPU_KERNEL_TWO(
  NeighborExchange,
  KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat32),
  NcclP2PGpuKernel, half, float)
}  // namespace kernel
}  // namespace mindspore
