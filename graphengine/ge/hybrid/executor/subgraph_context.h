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

#ifndef GE_HYBRID_EXECUTOR_ITERATION_CONTEXT_H_
#define GE_HYBRID_EXECUTOR_ITERATION_CONTEXT_H_

#include <vector>
#include "mmpa/mmpa_api.h"
#include "hybrid/common/tensor_value.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/executor/node_state.h"
#include "hybrid/executor/node_done_manager.h"
#include "hybrid/model/graph_item.h"
#include "hybrid/model/node_item.h"

namespace ge {
namespace hybrid {
class SubgraphContext {
 public:
  explicit SubgraphContext(const GraphItem *graph_item, GraphExecutionContext *execution_context);
  ~SubgraphContext();

  Status Init();
  void SetGroup(int group);
  void ResetContext(const NodePtr &node);
  void Reset();
  NodeStatePtr GetOrCreateNodeState(const NodeItem *node_item);

  void OnError(Status error);

  Status SetInput(const NodeItem &node_item, int input_index, const TensorValue &tensor);
  Status SetOutput(const NodeItem &node_item, int output_index, const TensorValue &tensor);
  Status SetInput(int index, const TensorValue &tensor);
  Status GetInput(int index, TensorValue &tensor);
  Status GetOutputs(std::vector<TensorValue> &outputs);

  Status Await(const NodePtr &node);
  void NodeDone(const NodePtr &node);

 private:
  NodeStatePtr CreateNodeState(const NodeItem *node_item);
  FrameStatePtr GetOrCreateFrameState(const NodeItem &node_item); // no lock
  friend class TaskContext;
  const GraphItem *graph_item_;
  GraphExecutionContext *execution_context_;
  mmRWLock_t rw_lock_;
  std::vector<TensorValue> all_inputs_;
  std::vector<TensorValue> all_outputs_;
  NodeDoneManager node_done_manager_;
  std::unordered_map<const NodeItem *, NodeStatePtr> node_states_;
  std::unordered_map<int64_t, FrameStatePtr> frame_states_;
  int group_ = -1;
};
}  // namespace hybrid
}  // namespace ge

#endif // GE_HYBRID_EXECUTOR_ITERATION_CONTEXT_H_
