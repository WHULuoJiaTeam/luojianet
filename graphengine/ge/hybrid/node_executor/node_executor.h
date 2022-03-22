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

#ifndef GE_HYBRID_NODE_EXECUTOR_NODE_EXECUTOR_H_
#define GE_HYBRID_NODE_EXECUTOR_NODE_EXECUTOR_H_

#include "external/ge/ge_api_error_codes.h"
#include "common/opskernel/ops_kernel_builder.h"
#include "graph/node.h"
#include "hybrid/node_executor/task_context.h"

namespace ge {
const uint32_t MEMORY_ALIGN_RATIO = 2;
const uint32_t MEMORY_ALIGN_SIZE = 32;
namespace hybrid {
class HybridModel;
using NodeTaskPtr = std::shared_ptr<NodeTask>;

// Base class of Node Task
class NodeTask {
 public:
  NodeTask() = default;
  virtual ~NodeTask() = default;

  /**
   * Update tiling data
   * @param context             instance of TaskContext
   * @return SUCCESS on success, error code otherwise
   */
  virtual Status UpdateTilingData(TaskContext &context) {
    return SUCCESS;
  }

  /**
   * Init
   * @param context             instance of TaskContext
   * @return SUCCESS on success, error code otherwise
   */
  virtual Status Init(TaskContext &context) {
    return SUCCESS;
  }

  /**
   * Whether this task supports dynamic shape
   * @return true if this task supports dynamic shape, false otherwise
   */
  virtual bool IsSupportDynamicShape() {
    return true;
  }

  /**
   * Update args for execution
   * @param context             instance of TaskContext
   * @return SUCCESS on success, error code otherwise
   */
  virtual Status UpdateArgs(TaskContext &context) = 0;

  /**
   * Execute task async
   * @param context             instance of TaskContext
   * @param done_callback       callback function, will be invoked after task is done
   * @return SUCCESS on success, error code otherwise
   */
  virtual Status ExecuteAsync(TaskContext &context, std::function<void()> done_callback) = 0;
};

class NoOpTask : public NodeTask {
 public:
  Status UpdateArgs(TaskContext &context) override;
  Status ExecuteAsync(TaskContext &context, std::function<void()> done_callback) override;
};

// Node executor
class NodeExecutor {
 public:
  NodeExecutor() = default;
  virtual ~NodeExecutor() = default;

  /**
   * Initialize node executor
   * @return SUCCESS on success, error code otherwise
   */
  virtual Status Initialize() {
    return SUCCESS;
  }

  /**
   * Finalize node executor
   * @return SUCCESS on success, error code otherwise
   */
  virtual Status Finalize() {
    return SUCCESS;
  }

  /**
   * Load task in load stage
   * @param model       instance of HybridModel
   * @param node        node
   * @param task        generated node task
   * @return SUCCESS on success, error code otherwise
   */
  virtual Status LoadTask(const HybridModel &model,
                          const NodePtr &node,
                          std::shared_ptr<NodeTask> &task) const;

  /**
   * Compile task in run stage
   * @param model       instance of HybridModel
   * @param node        node
   * @param task        generated node task
   * @return SUCCESS on success, error code otherwise
   */
  virtual Status CompileTask(const HybridModel &model,
                             const NodePtr &node,
                             std::shared_ptr<NodeTask> &task) const;

  /**
   * Preparation actions before execution
   * @param task        instance of NodeTask
   * @param context     instance of TaskContext
   * @return SUCCESS on success, error code otherwise
   */
  virtual Status PrepareTask(NodeTask &task, TaskContext &context) const;

  /**
   * Execute task
   * @param task        instance of NodeTask
   * @param context     instance of TaskContext
   * @param callback    callback function which will be invoked after computation is done
   * @return SUCCESS on success, error code otherwise
   */
  virtual Status ExecuteTask(NodeTask &task, TaskContext &context, const std::function<void()> &callback) const;
};

class NodeExecutorManager {
 public:
  enum class ExecutorType {
    AICORE,
    AICPU_TF,
    AICPU_CUSTOM,
    COMPILED_SUBGRAPH,
    DYNAMIC_SUBGRAPH,
    GE_LOCAL,
    CONTROL_OP,
    HCCL,
    RTS,
    HOST_CPU,
    RESERVED
  };

  static NodeExecutorManager &GetInstance() {
    static NodeExecutorManager instance;
    return instance;
  }

  /**
   * Register build of executor
   * @param executor_type   type of executor
   * @param builder         build function
   */
  void RegisterExecutorBuilder(ExecutorType executor_type, const std::function<NodeExecutor *()> &builder);

  /**
   * Initialize executor if needed
   * @return SUCCESS on success, error code otherwise
   */
  Status EnsureInitialized();

  void FinalizeExecutors();

  /**
   * CalcOpRunningParam
   * @param node        node
   * @return SUCCESS on success, error code otherwise
   */
  Status CalcOpRunningParam(Node &node) const;

  /**
   * Get executor by node
   * @param node            node
   * @param executor        executor
   * @return SUCCESS on success, error code otherwise
   */
  Status GetExecutor(Node &node, const NodeExecutor **executor);

  /**
   * Resolve executor type by node
   * @param node            node
   * @return executor type
   */
  ExecutorType ResolveExecutorType(Node &node) const;

  Status GetOrCreateExecutor(ExecutorType executor_type, const NodeExecutor **executor);

  bool IsExecutorInitialized(ExecutorType executor_type);

 private:
  std::map<ExecutorType, std::unique_ptr<NodeExecutor>> executors_;
  std::map<ExecutorType, std::function<NodeExecutor *()>> builders_;
  std::map<std::string, NodeExecutorManager::ExecutorType> engine_mapping_;
  std::mutex mu_;
  bool initialized_ = false;
  int ref_count_ = 0;
};

class NodeExecutorRegistrar {
 public:
  NodeExecutorRegistrar(NodeExecutorManager::ExecutorType executor_type,
                        NodeExecutor *(*builder)());
  ~NodeExecutorRegistrar() = default;
};
}  // namespace hybrid
}  // namespace ge

#define REGISTER_NODE_EXECUTOR_BUILDER(engine_type, executor) \
    REGISTER_NODE_EXECUTOR_BUILDER_UNIQ_HELPER(__COUNTER__, engine_type, executor)

#define REGISTER_NODE_EXECUTOR_BUILDER_UNIQ_HELPER(ctr, engine_type, executor) \
    REGISTER_NODE_EXECUTOR_BUILDER_UNIQ(ctr, engine_type, executor)

#define REGISTER_NODE_EXECUTOR_BUILDER_UNIQ(ctr, engine_type, executor)                         \
  static ::ge::hybrid::NodeExecutorRegistrar register_##ctr                                     \
      __attribute__((unused)) =                                                                 \
          ::ge::hybrid::NodeExecutorRegistrar(engine_type, []()->::ge::hybrid::NodeExecutor* {  \
            return new (std::nothrow) executor();                                               \
          })

#endif // GE_HYBRID_NODE_EXECUTOR_NODE_EXECUTOR_H_
