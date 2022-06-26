/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ASCEND_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ASCEND_KERNEL_MOD_H_

#include <vector>
#include <memory>
#include "plugin/device/ascend/hal/device/ge_runtime/task_info.h"
#include "kernel/kernel.h"
#include "runtime/device/executor/dynamic_kernel.h"
#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#endif

using TaskInfoPtr = std::shared_ptr<mindspore::ge::model_runner::TaskInfo>;
namespace mindspore {
namespace kernel {
class AscendKernelMod : public KernelMod {
 public:
  AscendKernelMod() = default;
  explicit AscendKernelMod(const AnfNodePtr &anf_node_ptr) : KernelMod(anf_node_ptr) {}
  virtual std::vector<TaskInfoPtr> GenTask(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                           const std::vector<AddressPtr> &, uint32_t) = 0;
  uint32_t block_dim() const { return block_dim_; }
  uint32_t stream_id() const { return stream_id_; }
  virtual bool NeedDump() {
#ifndef ENABLE_SECURITY
    const auto &dump_json = DumpJsonParser::GetInstance();
    return dump_json.NeedDump(fullname_) && dump_json.async_dump_enabled() && dump_json.op_debug_mode() == 0 &&
           !is_monad_;
#else
    return false;
#endif
  }
  void UpdateOp() override;
  bool IsNeedUpdateOp() override;

  void InitDynamicKernel(const CNodePtr &cnode_ptr, void *stream) {
    if (dynamic_kernel_ == nullptr) {
      stream_ = stream;
      dynamic_kernel_ = GenDynamicKernel(cnode_ptr, stream);
      dynamic_kernel_->Initialize();
    }
  }
  device::DynamicKernelPtr DynamicKernel() const { return dynamic_kernel_; }

 protected:
  uint32_t block_dim_{1};
  uint32_t stream_id_{0};
  device::DynamicKernelPtr dynamic_kernel_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ASCEND_KERNEL_MOD_H_
