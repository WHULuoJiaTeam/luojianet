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

#ifndef GE_HYBRID_EXECUTOR_NODE_STATE_H_
#define GE_HYBRID_EXECUTOR_NODE_STATE_H_

#include <condition_variable>
#include <future>
#include <mutex>

#include "common/blocking_queue.h"
#include "external/ge/ge_api_error_codes.h"
#include "hybrid/model/node_item.h"
#include "hybrid/executor/node_done_manager.h"

namespace ge {
namespace hybrid {
class NodeTask;
struct GraphExecutionContext;
class SubgraphContext;
class TaskContext;
struct NodeState;
struct FrameState;

using NodeStatePtr = std::shared_ptr<NodeState>;
using FrameStatePtr = std::shared_ptr<FrameState>;

class ShapeFuture {
 public:
  ShapeFuture(NodeState *src_node, uint32_t src_index, SubgraphContext *subgraph_context);
  ~ShapeFuture() = default;
  Status Get(GeShape &ori_shape, GeShape &shape);
  Status GetTensorDesc(const GeTensorDesc **tensor_desc);

 private:
  NodeState *src_node_;
  uint32_t src_index_;
  SubgraphContext *subgraph_context_;
};

struct ShapeInferenceState {
  explicit ShapeInferenceState(const NodeItem &node_item);

  void InitShapeState();

  Status UpdateInputShape(int idx, const GeTensorDesc &tensor_desc);

  void UpdateInputShapeFuture(int idx, ShapeFuture &&future);

  Status AwaitShapesReady(const GraphExecutionContext &context);

  Status UpdateOutputDesc();

  const vector<GeTensorDesc> &GetOutputTensorDesc() const;

  const NodeItem &node_item;

 private:
  Status UpdateInputForMerge(const GraphExecutionContext &context);

  friend struct NodeState;
  std::vector<std::pair<int, ShapeFuture>> shape_futures;
  // do not directly update op_desc, in case race condition across pipelines
  std::vector<GeTensorDesc> input_tensor_desc;
  std::vector<GeTensorDesc> output_tensor_desc;

  int num_pending_shapes_ = 0;
  std::condition_variable ready_cv_;
  std::mutex mu_;
};

struct FrameState {
 public:
  FrameState(int64_t id) : frame_id_(id) {}
  ~FrameState() = default;

  int64_t frame_id_{0};
  uint64_t active_count_{0};
  uint64_t iteration_count_{0};

  std::shared_ptr<FrameState> parent_frame_;
};

// saving sth. dynamic during execution
struct NodeState {
 public:
  NodeState(const NodeItem &node_item, SubgraphContext *subgraph_context);
  ~NodeState() = default;

  Status Init(int group, const shared_ptr<FrameState> &frame_state);

  OpDesc *GetOpDesc() const {
    return op_desc_.get();
  }

  inline const NodeItem *GetNodeItem() const {
    return node_item_;
  }

  inline const string &GetName() const {
    return node_item_->NodeName();
  }

  inline const string &GetType() const {
    return node_item_->NodeType();
  }

  ShapeInferenceState &GetShapeInferenceState() {
    return shape_inference_state_;
  }

  Status UpdateOutputShapes(int index, const GeShape &shape, const GeShape &ori_shape);

  inline bool IsShapeDependence() const {
    return node_item_->IsControlFlowOp() || node_item_->shape_inference_type >= DEPEND_SHAPE_RANGE;
  }

  void RunStreamActive();
  void RunNextIteration();

  void SavePersistTensor(int input_idx, const TensorValue &tensor);
  void UpdatePersistTensor();

  Status NodeScheduled(const std::function<void(const NodeItem *)> &ready) const;

  void SetScheduleFuture(std::future<Status> &&future);
  Status WaitForScheduleDone();

  void SetSwitchIndex(int index) {
    switch_index_ = index;
  }

  int GetSwitchIndex() const {
    return switch_index_;
  }

  void SetMergeIndex(int index) {
    merge_index_ = index;
  }

  int GetMergeIndex() const {
    return merge_index_;
  }

  int GetGroup() const {
    return group_;
  }

  const shared_ptr<NodeTask> &GetKernelTask() const {
    return kernel_task_;
  }

  void SetKernelTask(const shared_ptr<NodeTask> &kernel_task) {
    kernel_task_ = kernel_task;
  }

  Status WaitForPrepareDone();

  void SetPrepareFuture(std::future<Status> &&prepare_future) {
    this->prepare_future_ = std::move(prepare_future);
  }

  Status AwaitInputTensors(GraphExecutionContext &context) const;

  void SetTaskContext(std::shared_ptr<TaskContext> &task_context);
  std::shared_ptr<TaskContext> GetTaskContext();

  void SetSkipInferShape(bool skip_infershape) { skip_infershape_ = skip_infershape; }

  bool MaySkipShapeInference() const { return skip_infershape_; }

 private:
  bool IsScheduleReady() const;
  void SetDataSchedule(const NodeState &node_state, const std::function<void(const NodeItem *)> &ready);
  void SetCtrlSchedule(const NodeState &node_state, const std::function<void(const NodeItem *)> &ready);
  void ResetContext(uint64_t iteration);
  void ScheduleContext(const NodeState &node_state);
  void UpdatePersistTensor(int input_idx);

  const NodeItem *node_item_ = nullptr;
  std::shared_ptr<NodeTask> kernel_task_ = nullptr;
  std::future<Status> prepare_future_;
  OpDescPtr op_desc_;
  ShapeInferenceState shape_inference_state_;
  SubgraphContext *subgraph_context_;
  std::shared_ptr<TaskContext> task_context_ = nullptr;
  std::mutex mu_;

  std::future<Status> schedule_future_;
  std::shared_ptr<FrameState> frame_state_;
  std::map<int, TensorValue> root_tensor_values_;
  uint64_t active_count_ = 0;
  uint64_t iteration_count_ = 0;
  uint32_t ctrl_scheduled_ = 0;
  uint32_t data_scheduled_ = 0;
  int merge_index_ = -1; // Use for Execute (Reset after Executed).
  int switch_index_ = -1; // Use for Schedule (Reset after Prepared).
  int group_ = -1;
  bool skip_infershape_ = false;
};
}  // namespace hybrid
}  // namespace ge

#endif // GE_HYBRID_EXECUTOR_NODE_STATE_H_
