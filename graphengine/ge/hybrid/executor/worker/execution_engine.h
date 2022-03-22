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

#ifndef GE_HYBRID_EXECUTOR_EXECUTOR_EXECUTION_ENGINE_H_
#define GE_HYBRID_EXECUTOR_EXECUTOR_EXECUTION_ENGINE_H_

#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/node_executor/task_context.h"
#include "common/dump/dump_op.h"

namespace ge {
namespace hybrid {
class NodeDoneCallback {
 public:
  NodeDoneCallback(GraphExecutionContext *graph_context, std::shared_ptr<TaskContext> task_context);
  ~NodeDoneCallback() = default;
  Status OnNodeDone();
 private:
  Status PrepareConstInputs(const NodeItem &node_item);
  Status DumpDynamicNode();
  Status ProfilingReport();
  Status SaveDumpOpInfo();
  Status GetTaskDescInfo(const NodePtr node, const HybridModel *model,
                         std::vector<TaskDescInfo> &task_desc_info);
  GraphExecutionContext *graph_context_;
  std::shared_ptr<TaskContext> context_;
  DumpOp dump_op_;
};

class ExecutionEngine {
 public:
  static Status ExecuteAsync(NodeState &node_state,
                             const std::shared_ptr<TaskContext> &task_context,
                             GraphExecutionContext &execution_context,
                             const std::function<void()> &callback);

 private:
  static Status ValidateInputTensors(const NodeState &node_state, const TaskContext &task_context);
  static Status PropagateOutputs(const NodeItem &node_item, TaskContext &task_context, GraphExecutionContext &context);
  static Status DoExecuteAsync(NodeState &node_state,
                               TaskContext &task_context,
                               GraphExecutionContext &context,
                               const std::function<void()> &callback);
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_EXECUTOR_EXECUTOR_EXECUTION_ENGINE_H_
