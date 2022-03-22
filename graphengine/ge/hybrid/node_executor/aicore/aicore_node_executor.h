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

#ifndef GE_HYBRID_KERNEL_AICORE_NODE_EXECUTOR_H_
#define GE_HYBRID_KERNEL_AICORE_NODE_EXECUTOR_H_

#include "hybrid/node_executor/aicore/aicore_task_builder.h"
#include "hybrid/node_executor/node_executor.h"
#include <map>
#include <mutex>

namespace ge {
namespace hybrid {
class TaskCompiler {
 public:
  TaskCompiler() = default;
  virtual ~TaskCompiler() = default;
  virtual Status CompileOp(const NodePtr &node, std::vector<domi::TaskDef> &tasks) = 0;
  virtual Status Initialize() = 0;
};

class AiCoreNodeTaskRegistry {
 public:
  ~AiCoreNodeTaskRegistry() = default;

  static AiCoreNodeTaskRegistry &GetInstance() {
    static AiCoreNodeTaskRegistry instance;
    return instance;
  }

  std::shared_ptr<AiCoreNodeTask> GetTask(const std::string &node_key);
  bool AddTask(const std::string &node_key, const std::shared_ptr<AiCoreNodeTask> &task);
 private:
  AiCoreNodeTaskRegistry() = default;
  std::map<std::string, std::shared_ptr<AiCoreNodeTask>> reg_node_tasks_;
  std::mutex mutex_;
};

class AiCoreNodeTask : public NodeTask {
 public:
  explicit AiCoreNodeTask(std::vector<std::unique_ptr<AiCoreOpTask>> &&tasks);
  ~AiCoreNodeTask() override = default;
  bool IsSupportDynamicShape() override;
  Status UpdateTilingData(TaskContext &context) override;

  Status UpdateArgs(TaskContext &context) override;
  Status ExecuteAsync(TaskContext &context, std::function<void()> done_callback) override;

  const vector<int64_t> &GetWorkspaceSizes() const;
  void SetWorkspaceSizes(const vector<int64_t> &workspace_sizes);
 private:
  Status CheckOverflow(TaskContext &context);
  std::vector<std::unique_ptr<AiCoreOpTask>> tasks_;
  std::vector<int64_t> workspace_sizes_;
};

class AiCoreNodeExecutor : public NodeExecutor {
 public:
  Status Initialize() override;
  Status LoadTask(const HybridModel &model, const NodePtr &node, shared_ptr<NodeTask> &task) const override;
  Status CompileTask(const HybridModel &model, const NodePtr &node,
                     std::shared_ptr<NodeTask> &task) const override;

 private:
  static Status GenNodeKey(const NodePtr &node, std::string &node_key);
  std::unique_ptr<TaskCompiler> compiler_;
};

using CreateFn = TaskCompiler *(*)();
class TaskCompilerFactory {
 public:
  static TaskCompilerFactory &GetInstance();
  void Register(CreateFn fn);
  std::unique_ptr<TaskCompiler> GetTaskCompiler();

 private:
  CreateFn compiler_func_;
};

class CompilerFunctionRegistrar {
 public:
  explicit CompilerFunctionRegistrar(CreateFn fn);
  ~CompilerFunctionRegistrar() = default;
};
}  // namespace hybrid
}  // namespace ge

#define REGISTER_TASK_COMPILER(compiler)                                                         \
  static ::ge::hybrid::CompilerFunctionRegistrar register_compiler_function                      \
      __attribute__((unused)) =                                                                  \
          ::ge::hybrid::CompilerFunctionRegistrar([]()->::ge::hybrid::TaskCompiler* {            \
            return new (std::nothrow) compiler();                                                \
          })                                                                                     \

#endif  //GE_HYBRID_KERNEL_AICORE_NODE_EXECUTOR_H_
