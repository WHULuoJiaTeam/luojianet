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

#ifndef GE_HYBRID_NODE_EXECUTOR_RTS_RTS_NODE_EXECUTOR_H_
#define GE_HYBRID_NODE_EXECUTOR_RTS_RTS_NODE_EXECUTOR_H_

#include "hybrid/node_executor/node_executor.h"
#include "hybrid/node_executor/rts/rts_node_task.h"

namespace ge {
namespace hybrid {
class IdentityNodeTask : public RtsNodeTask {
 public:
  Status ExecuteAsync(TaskContext &context, std::function<void()> done_callback) override;

 protected:
  static Status DoCopyTensor(TaskContext &context, int index);
};

class IdentityNNodeTask : public IdentityNodeTask {
 public:
  Status ExecuteAsync(TaskContext &context, std::function<void()> done_callback) override;
};

class ReadVariableOpNodeTask : public IdentityNodeTask {
 public:
  Status ExecuteAsync(TaskContext &context, std::function<void()> done_callback) override;
};

class ProfilingTraceNodeTask :  public RtsNodeTask {
 public:
  Status Init(const HybridModel &model, const NodePtr &node) override;

  Status ExecuteAsync(TaskContext &context, std::function<void()> done_callback) override;

 private:
  std::vector<domi::TaskDef> task_defs_;
};

class RtsNodeExecutor : public NodeExecutor {
 public:
  Status LoadTask(const HybridModel &model, const NodePtr &node, shared_ptr<NodeTask> &task) const override;
};
}  // namespace hybrid
}  // namespace ge

#endif // GE_HYBRID_NODE_EXECUTOR_RTS_RTS_NODE_EXECUTOR_H_
