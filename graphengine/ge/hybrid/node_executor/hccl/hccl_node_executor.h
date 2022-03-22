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

#ifndef HYBRID_HCCL_NODE_EXECUTOR_H_
#define HYBRID_HCCL_NODE_EXECUTOR_H_
#include "common/opskernel/ge_task_info.h"
#include "graph/op_desc.h"
#include "graph/runtime_inference_context.h"
#include "hybrid/model/hybrid_model.h"
#include "hybrid/node_executor/node_executor.h"

namespace ge {
namespace hybrid {
class HybridModel;

class HcclNodeTask : public NodeTask {
 public:
  HcclNodeTask() {}

  ~HcclNodeTask() {}

  Status UpdateArgs(TaskContext &context) override;
  Status ExecuteAsync(TaskContext &context, std::function<void()> done_callback) override;
  Status Init(TaskContext &context) override;

 private:
  std::shared_ptr<DavinciModel> davinci_model_ = nullptr;
  bool load_flag_ = false;
  std::mutex hccl_mutex_;
  std::condition_variable cond_;
};

class RdmaNodeTask : public NodeTask {
 public:
  RdmaNodeTask() = default;

  ~RdmaNodeTask() override {}

  Status UpdateArgs(TaskContext &context) override;
  Status ExecuteAsync(TaskContext &context, std::function<void()> done_callback) override;
  Status Init(TaskContext &context) override;

 private:
  Status SetAddrInfo(TaskContext &context, RuntimeInferenceContext *ctx, uint64_t *data, int64_t row_num,
                     vector<HcomRemoteAccessAddrInfo> &addr_infos);
  Status ExtractTensor(TaskContext &context, vector<HcomRemoteAccessAddrInfo> &addr_infos);
  std::pair<int64_t, int64_t> remote_index_;
  std::pair<int64_t, int64_t> offset_index_;
  int32_t local_index_ = 0;
  std::mutex hccl_mutex_;
  std::condition_variable cond_;
  bool skip_flag_ = false;
};


class AllToAllNodeTask : public NodeTask {
 public:
  AllToAllNodeTask() = default;

  ~AllToAllNodeTask() = default;

  Status UpdateArgs(TaskContext &context) override { return SUCCESS; }
  Status ExecuteAsync(TaskContext &context, std::function<void()> done_callback) override;
  Status Init(TaskContext &context) override { return SUCCESS; }

 private:
  std::mutex hccl_mutex_;
  std::condition_variable cond_;
};

class HcclNodeExecutor : public NodeExecutor {
 public:
  Status LoadTask(const HybridModel &model, const NodePtr &node, shared_ptr<NodeTask> &task) const;
  Status PrepareTask(NodeTask &task, TaskContext &context) const;
  Status ExecuteTask(NodeTask &task, TaskContext &context, const std::function<void()> &callback) const;
  Status Initialize() override;
  Status Finalize() override;
  ~HcclNodeExecutor() {}

 private:
  void *handle_;
};
}  // namespace hybrid
}  // namespace ge

#endif  // HYBRID_HCCL_NODE_EXECUTOR_H_
