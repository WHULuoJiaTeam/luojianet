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

#ifndef GE_HYBRID_KERNEL_AICORE_TASK_BUILDER_H_
#define GE_HYBRID_KERNEL_AICORE_TASK_BUILDER_H_

#include <vector>
#include <string>
#include "hybrid/node_executor/aicore/aicore_op_task.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/attr_utils.h"
#include "graph/op_kernel_bin.h"
#include "proto/task.pb.h"

namespace ge {
namespace hybrid {
class AiCoreNodeTask;
class AiCoreKernelRegistry {
 public:
  ~AiCoreKernelRegistry() = default;
  static AiCoreKernelRegistry &GetInstance() {
    static AiCoreKernelRegistry instance;
    return instance;
  }
  const char *GetUnique(const string &stub_func);

 private:
  AiCoreKernelRegistry() = default;
  std::set<std::string> unique_stubs_;
  std::mutex mutex_;
};

class AiCoreTaskBuilder {
 public:
  AiCoreTaskBuilder(const OpDescPtr &op_desc, const std::vector<domi::TaskDef> &task_defs);
  ~AiCoreTaskBuilder() = default;

  Status BuildTask(std::unique_ptr<AiCoreNodeTask> &node_task,
                   bool ignore_failure_on_atomic,
                   bool is_single_op = false);

 private:
  bool ExpectAtomicAddrCleanTask();

  OpDescPtr op_desc_;
  const std::vector<domi::TaskDef> &task_defs_;
};
}  // namespace hybrid
}  // namespace ge
#endif //GE_HYBRID_KERNEL_AICORE_TASK_BUILDER_H_
