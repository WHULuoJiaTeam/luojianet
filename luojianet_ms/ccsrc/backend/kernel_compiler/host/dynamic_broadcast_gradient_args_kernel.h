/**
 * Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022 Huawei Technologies Co., Ltd
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
#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_HOST_DYNAMIC_BROADCAST_GRADIENT_ARGS_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_HOST_DYNAMIC_BROADCAST_GRADIENT_ARGS_KERNEL_H_
#include <vector>
#include <memory>
#include <string>
#include "runtime/device/ascend/executor/host_dynamic_kernel.h"
#include "backend/kernel_compiler/host/host_kernel_mod.h"

using HostDynamicKernel = luojianet_ms::device::ascend::HostDynamicKernel;
namespace luojianet_ms {
namespace kernel {
class DynamicBroadcastGradientArgsKernel : public HostDynamicKernel {
 public:
  DynamicBroadcastGradientArgsKernel(void *stream, const CNodePtr &cnode_ptr) : HostDynamicKernel(stream, cnode_ptr) {}
  ~DynamicBroadcastGradientArgsKernel() override = default;
  void Execute() override;
};

class DynamicBroadcastGradientArgsKernelMod : public HostKernelMod {
 public:
  DynamicBroadcastGradientArgsKernelMod() = default;
  ~DynamicBroadcastGradientArgsKernelMod() override = default;
  device::DynamicKernelPtr GenDynamicKernel(const CNodePtr &cnode_ptr, void *stream_ptr) override;
};
MS_HOST_REG_KERNEL(DynamicBroadcastGradientArgs, DynamicBroadcastGradientArgsKernelMod);
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_HOST_DYNAMIC_BROADCAST_GRADIENT_ARGS_KERNEL_H_
