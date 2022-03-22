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

#ifndef GE_HYBRID_KERNEL_GE_LOCAL_NODE_EXECUTOR_H_
#define GE_HYBRID_KERNEL_GE_LOCAL_NODE_EXECUTOR_H_

#include <unordered_map>
#include <vector>
#include "hybrid/node_executor/node_executor.h"

namespace ge {
namespace hybrid {
class RefInputTask : public NodeTask {
 public:
  explicit RefInputTask(const NodePtr &node)
      : node_name_(node->GetName()),
        node_type_(node->GetType()) {
  }

  ~RefInputTask() = default;

  virtual Status UpdateArgs(TaskContext &context) override;
  virtual Status ExecuteAsync(TaskContext &context, std::function<void()> done_callback) override;
  static bool IsBelong(const std::string &op_type);
 private:
  Status Execute(TaskContext &context);
  Status RefOneByOne(TaskContext &context);
  Status RefByOrder(const std::vector<uint32_t> &ref_order, TaskContext &context);

 private:
  const std::string node_name_;
  const std::string node_type_;

  // key is op type, value is output ref input index,
  // e.g. {1,0} means out[0] ref input[1], out[1] ref input[0], if vector is empty, it means ref input one by one
  static const std::map<std::string, std::vector<uint32_t>> out_ref_input_index_;
};

class DependInputShapeTask : public NodeTask {
 public:
  explicit DependInputShapeTask(const NodePtr &node) : node_(node) {
  }

  ~DependInputShapeTask() = default;

  virtual Status UpdateArgs(TaskContext &context) override;
  virtual Status ExecuteAsync(TaskContext &context, std::function<void()> done_callback) override;
  static bool IsBelong(const std::string &op_type);
 private:
  Status Execute(TaskContext &context);
 private:
  const NodePtr node_;

  // ops depend input shape
  static const std::set<std::string> depend_input_shape_ops_;
};

class ConstantNodeTask : public NodeTask {
 public:
  explicit ConstantNodeTask(const TensorValue *tensor);
  ~ConstantNodeTask() = default;
  Status UpdateArgs(TaskContext &context) override;

  Status ExecuteAsync(TaskContext &context, std::function<void()> done_callback) override;
  static bool IsBelong(const std::string &op_type);

 private:
  static const std::set<std::string> constant_like_task_ops_;
  const TensorValue *tensor_;
};

class NoOpNodeTask : public NodeTask {
 public:
  explicit NoOpNodeTask() = default;
  ~NoOpNodeTask() = default;
  Status UpdateArgs(TaskContext &context) override;
  Status ExecuteAsync(TaskContext &context, std::function<void()> done_callback) override;
  static bool IsBelong(const std::string &op_type);

 private:
  static const std::set<std::string> control_only_task_ops_;
};

class GeLocalNodeExecutor : public NodeExecutor {
 public:

  Status PrepareTask(NodeTask &task, TaskContext &context) const override;

  virtual Status LoadTask(const HybridModel &model,
                          const NodePtr &node,
                          std::shared_ptr<NodeTask> &task) const override;
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_KERNEL_GE_LOCAL_NODE_EXECUTOR_H_
