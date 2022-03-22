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

#ifndef GE_HYBRID_KERNEL_AICORE_TASK_COMPILER_H_
#define GE_HYBRID_KERNEL_AICORE_TASK_COMPILER_H_

#include <mutex>
#include "opskernel_manager/ops_kernel_manager.h"
#include "hybrid/node_executor/aicore/aicore_node_executor.h"

namespace ge {
namespace hybrid {
class AiCoreTaskCompiler : public TaskCompiler {
 public:
  AiCoreTaskCompiler() = default;
  ~AiCoreTaskCompiler() override = default;

  Status CompileOp(const NodePtr &node, std::vector<domi::TaskDef> &tasks) override;
  Status Initialize() override;
 private:
  Status DoCompileOp(const NodePtr &node) const;
  Status DoGenerateTask(const Node &node, std::vector<domi::TaskDef> &tasks);
  OpsKernelInfoStorePtr aic_kernel_store_;
  bool is_initialized_ = false;
  static std::mutex mu_;
};
}  // namespace hybrid
}  // namespace ge
#endif //GE_HYBRID_KERNEL_AICORE_TASK_COMPILER_H_
