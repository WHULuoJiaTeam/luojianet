/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HOST_DYNAMIC_SHAPE_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HOST_DYNAMIC_SHAPE_KERNEL_H_
#include <vector>
#include <memory>
#include <string>
#include "plugin/device/ascend/hal/device/executor/host_dynamic_kernel.h"
#include "plugin/device/ascend/kernel/host/host_kernel_mod.h"
#include "kernel/kernel.h"
using HostDynamicKernel = mindspore::device::ascend::HostDynamicKernel;
namespace mindspore {
namespace kernel {
class TensorShapeKernel : public HostDynamicKernel {
 public:
  TensorShapeKernel(void *stream, const CNodePtr &cnode_ptr) : HostDynamicKernel(stream, cnode_ptr) {}
  ~TensorShapeKernel() override = default;
  void Execute() override;
  void Execute(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
};

class TensorShapeKernelMod : public HostKernelMod {
 public:
  TensorShapeKernelMod() = default;
  ~TensorShapeKernelMod() override = default;
  device::DynamicKernelPtr GenDynamicKernel(const CNodePtr &cnode_ptr, void *stream_ptr) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
};
MS_HOST_REG_KERNEL(DynamicShape, TensorShapeKernelMod);
MS_HOST_REG_KERNEL(TensorShape, TensorShapeKernelMod);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HOST_DYNAMIC_SHAPE_KERNEL_H_
