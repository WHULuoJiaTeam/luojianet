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

#ifndef GE_HYBRID_CONTROLOP_CONTROL_OP_EXECUTOR_H_
#define GE_HYBRID_CONTROLOP_CONTROL_OP_EXECUTOR_H_

#include <vector>
#include "hybrid/node_executor/node_executor.h"
#include "hybrid/model/graph_item.h"

namespace ge {
namespace hybrid {
class ControlOpNodeTask : public NodeTask {
 public:
  using NodeTask::Init;
  virtual Status Init(const NodePtr &node, const HybridModel &model) = 0;
  Status UpdateArgs(TaskContext &context) override;

  Status ExecuteAsync(TaskContext &task_context, std::function<void()> done_callback) override;

 protected:
  virtual Status DoExecuteAsync(TaskContext &task_context, const std::function<void()> &done_callback) const = 0;
  static Status ToBool(const TensorValue &tensor_value, DataType data_type, bool &value);
  static Status ExecuteSubgraph(const GraphItem *subgraph,
                                TaskContext &task_context,
                                const std::function<void()> &done_callback);
};

class IfOpNodeTask : public ControlOpNodeTask {
 public:
  Status Init(const NodePtr &node, const HybridModel &model) override;

 protected:
  Status DoExecuteAsync(TaskContext &task_context, const std::function<void()> &done_callback) const override;

 private:
  static constexpr int kIfCondIndex = 0;
  static constexpr int kThenBranchIndex = 0;
  static constexpr int kElseBranchIndex = 1;

  const GraphItem *then_ = nullptr;
  const GraphItem *else_ = nullptr;
};

class CaseOpNodeTask : public ControlOpNodeTask {
 public:
  Status Init(const NodePtr &node, const HybridModel &model) override;

 protected:
  const GraphItem* SelectBranch(int32_t branch_index) const;
  Status DoExecuteAsync(TaskContext &task_context, const std::function<void()> &done_callback) const override;

 private:
  static constexpr int kCaseBranchIndex = 0;
  static constexpr size_t kMaxBranchNum = INT32_MAX;
  static constexpr size_t kMinBranchNum = 1;

  std::vector<const GraphItem *> subgraphs_;
};

class WhileOpNodeTask : public ControlOpNodeTask {
 public:
  Status Init(const NodePtr &node, const HybridModel &model) override;

 protected:
  Status DoExecuteAsync(TaskContext &task_context, const std::function<void()> &done_callback) const override;
  Status ExecuteCond(TaskContext &task_context, bool &is_continue) const;

  static Status MoveOutputs2Inputs(TaskContext &task_context);
  Status ExecuteOneLoop(TaskContext &task_context, bool &is_continue) const;

 private:
  static constexpr int kCondBranchIndex = 0;
  static constexpr int kBodyBranchIndex = 1;
  static constexpr size_t kCondOutputSize = 1;

  const GraphItem *cond_ = nullptr;
  const GraphItem *body_ = nullptr;
};

class ControlOpNodeExecutor : public NodeExecutor {
 public:
  Status LoadTask(const HybridModel &model, const NodePtr &node, shared_ptr<NodeTask> &task) const override;
  Status PrepareTask(NodeTask &task, TaskContext &context) const override;
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_CONTROLOP_CONTROL_OP_EXECUTOR_H_
