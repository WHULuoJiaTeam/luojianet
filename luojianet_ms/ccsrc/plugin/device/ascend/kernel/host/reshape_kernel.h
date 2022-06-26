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
#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_HOST_RESHAPE_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_HOST_RESHAPE_KERNEL_H_
#include <vector>
#include <memory>
#include <string>
#include "plugin/device/ascend/hal/device/executor/host_dynamic_kernel.h"
#include "plugin/device/ascend/kernel/host/host_kernel_mod.h"
using HostDynamicKernel = luojianet_ms::device::ascend::HostDynamicKernel;
namespace luojianet_ms {
namespace kernel {
class ReshapeKernel : public HostDynamicKernel {
 public:
  ReshapeKernel(void *stream, const CNodePtr &cnode_ptr) : HostDynamicKernel(stream, cnode_ptr) {}
  ~ReshapeKernel() override = default;
  void Execute() override;
  void Execute(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
};

class ReshapeKernelMod : public HostKernelMod {
 public:
  ReshapeKernelMod() = default;
  ~ReshapeKernelMod() override = default;
  device::DynamicKernelPtr GenDynamicKernel(const CNodePtr &cnode_ptr, void *stream_ptr) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto node = anf_node_.lock();
    MS_EXCEPTION_IF_NULL(node);
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (kernel_ == nullptr) {
      kernel_ = std::dynamic_pointer_cast<ReshapeKernel>(GenDynamicKernel(cnode, stream_ptr));
      kernel_->Initialize();
    }
    if (stream_ == nullptr) {
      stream_ = stream_ptr;
    }
    try {
      kernel_->Execute(inputs, outputs);
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "ReshapeKernelMod Launch failed. node: " << cnode->fullname_with_scope() << ", Error message is "
                    << e.what();
      return false;
    }
    return true;
  }
  void UpdateOp() override { AscendKernelMod::UpdateOp(); }

 private:
  std::shared_ptr<ReshapeKernel> kernel_;
};
MS_HOST_REG_KERNEL(Reshape, ReshapeKernelMod);
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_HOST_RESHAPE_KERNEL_H_
