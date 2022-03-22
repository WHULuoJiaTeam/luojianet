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

#ifndef GE_HYBRID_KERNEL_HOST_CPU_NODE_EXECUTOR_H_
#define GE_HYBRID_KERNEL_HOST_CPU_NODE_EXECUTOR_H_

#include "hybrid/node_executor/node_executor.h"
#include "inc/kernel.h"
#include "hybrid/node_executor/aicpu/aicpu_node_executor.h"

namespace ge {
namespace hybrid {

class HostAicpuNodeTask : public AicpuNodeTask {
 public:
  HostAicpuNodeTask(const NodeItem *node_item, const domi::TaskDef &task_def)
      : AicpuNodeTask(node_item, task_def) {}
  ~HostAicpuNodeTask() override = default;

  Status ExecuteAsync(TaskContext &context, std::function<void()> done_callback) override;

  Status UpdateArgs(TaskContext &context) override;

  void SetRunKernel(std::function<uint32_t(void *)> run_cpu_kernel) { run_cpu_kernel_ = run_cpu_kernel; }

  Status SetHostExtInfo();

 private:
  Status Execute(TaskContext &context);

  std::function<uint32_t(void *)> run_cpu_kernel_ = nullptr;
};

class HostCpuNodeExecutor : public NodeExecutor {
 public:
  Status PrepareTask(NodeTask &task, TaskContext &context) const override;

  Status LoadTask(const HybridModel &model,
                  const NodePtr &node,
                  std::shared_ptr<NodeTask> &task) const override;

 private:
  static Status ValidateTaskDef(const domi::TaskDef &task_def);
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_KERNEL_HOST_CPU_NODE_EXECUTOR_H_
