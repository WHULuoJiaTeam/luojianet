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

#ifndef GE_HYBRID_NODE_EXECUTOR_RTS_RTS_NODE_TASK_H_
#define GE_HYBRID_NODE_EXECUTOR_RTS_RTS_NODE_TASK_H_

#include "hybrid/node_executor/node_executor.h"
#include "proto/task.pb.h"

namespace ge {
namespace hybrid {
class RtsNodeTask : public NodeTask {
 public:
  Status Init(TaskContext &task_context) override {
    return SUCCESS;
  }

  virtual Status Init(const HybridModel &model, const NodePtr &node) {
    GELOGD("[%s] Done initialization successfully.", node->GetName().c_str());
    return SUCCESS;
  }

  Status UpdateArgs(TaskContext &task_context) override {
    GELOGD("[%s] Done update args successfully.", task_context.GetNodeName());
    return SUCCESS;
  }

  static Status GetScalarIndexValue(TaskContext &task_context, uint32_t index, int64_t &value);
};

class StreamActiveNodeTask : public RtsNodeTask {
 public:
  Status ExecuteAsync(TaskContext &task_context, std::function<void()> done_callback) override;
};

class StreamSwitchNodeTask : public RtsNodeTask {
 public:
  Status Init(const HybridModel &model, const NodePtr &node) override;
  Status ExecuteAsync(TaskContext &task_context, std::function<void()> done_callback) override;

 private:
  std::function<bool(int64_t, int64_t)> comp_func_{nullptr};
};

class StreamMergeNodeTask : public RtsNodeTask {
 public:
  Status ExecuteAsync(TaskContext &task_context, std::function<void()> done_callback) override;
};

class PassThroughNodeTask : public RtsNodeTask {
 public:
  Status ExecuteAsync(TaskContext &task_context, std::function<void()> done_callback) override;
};

class LabelSetNodeTask : public RtsNodeTask {
 public:
  Status ExecuteAsync(TaskContext &task_context, std::function<void()> done_callback) override;
};

class LabelGotoNodeTask : public RtsNodeTask {
 public:
  Status ExecuteAsync(TaskContext &task_context, std::function<void()> done_callback) override;
};

class LabelSwitchNodeTask : public RtsNodeTask {
 public:
  Status ExecuteAsync(TaskContext &task_context, std::function<void()> done_callback) override;
};
}  // namespace hybrid
}  // namespace ge
#endif  // GE_HYBRID_NODE_EXECUTOR_RTS_RTS_NODE_TASK_H_
