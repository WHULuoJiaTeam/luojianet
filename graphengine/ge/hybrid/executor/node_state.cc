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

#include "hybrid/executor/node_state.h"
#include <chrono>
#include "framework/common/debug/log.h"
#include "graph/compute_graph.h"
#include "graph/utils/tensor_utils.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/executor/subgraph_context.h"
#include "hybrid/node_executor/task_context.h"

#define INC_ITERATION_COUNT(iteration) \
do {                                   \
  ++iteration;                         \
  if (iteration == UINT64_MAX) {       \
    iteration = 1;                     \
  }                                    \
} while (0)

namespace ge {
namespace hybrid {
namespace {
// 5s * 120, wait for 10m
constexpr auto kWaitInternal = 5;
constexpr auto kMaxWaitTimes = 120;
}
ShapeInferenceState::ShapeInferenceState(const NodeItem &node_item) : node_item(node_item) {
  InitShapeState();
}

void ShapeInferenceState::InitShapeState() {
  this->num_pending_shapes_ = node_item.num_inputs - node_item.num_static_input_shapes;
  GELOGD("[%s] ShapeInferenceState created, pending shape count = %d",
         node_item.NodeName().c_str(),
         this->num_pending_shapes_);

  input_tensor_desc.resize(node_item.num_inputs);
  for (int i = 0; i < node_item.num_inputs; ++i) {
    node_item.GetInputDesc(i, input_tensor_desc[i]);
  }

  output_tensor_desc.resize(node_item.num_outputs);
  for (int i = 0; i < node_item.num_outputs; ++i) {
    node_item.GetOutputDesc(i, output_tensor_desc[i]);
  }
}

Status ShapeInferenceState::UpdateInputShape(int idx, const GeTensorDesc &target) {
  if (node_item.IsInputShapeStatic(idx)) {
    GELOGD("[%s] Trying to update static shape, idx = %d. old shape = [%s], new shape = [%s]",
           node_item.NodeName().c_str(),
           idx,
           node_item.MutableInputDesc(idx)->GetShape().ToString().c_str(),
           target.GetShape().ToString().c_str());
    return SUCCESS;
  }

  std::lock_guard<std::mutex> lk(mu_);
  auto &input_desc = input_tensor_desc[idx];
  GeShape shape = target.GetShape();
  input_desc.SetShape(shape);
  input_desc.SetOriginShape(target.GetOriginShape());
  int64_t tensor_size = -1;
  (void) TensorUtils::GetSize(target, tensor_size);
  if (tensor_size <= 0) {
    Format format = input_desc.GetFormat();
    DataType data_type = input_desc.GetDataType();
    if (TensorUtils::CalcTensorMemSize(shape, format, data_type, tensor_size) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "[Invoke][CalcTensorMemSize] failed for [%s].", node_item.NodeName().c_str());
      REPORT_CALL_ERROR("E19999", "CalcTensorMemSize failed for [%s].", node_item.NodeName().c_str());
      return FAILED;
    }
  }
  GELOGD("[%s] Update input shape [%d] with Shape: [%s] and OriginalShape: [%s], size = %ld",
         node_item.NodeName().c_str(),
         idx,
         shape.ToString().c_str(),
         target.GetOriginShape().ToString().c_str(),
         tensor_size);
  (void) TensorUtils::SetSize(input_desc, tensor_size);
  if (--num_pending_shapes_ <= 0) {
    ready_cv_.notify_all();
  }

  return SUCCESS;
}

void ShapeInferenceState::UpdateInputShapeFuture(int idx, ShapeFuture &&future) {
  if (node_item.IsInputShapeStatic(idx)) {
    GELOGD("[%s] Trying to update constant shape, idx = %d", node_item.NodeName().c_str(), idx);
    return;
  }

  GELOGD("[%s] Update input shape [%d] with ShapeFuture.", node_item.NodeName().c_str(), idx);
  std::lock_guard<std::mutex> lk(mu_);
  shape_futures.emplace_back(idx, std::move(future));
  if (--num_pending_shapes_ == 0) {
    ready_cv_.notify_all();
  }
}

Status ShapeInferenceState::UpdateInputForMerge(const GraphExecutionContext &context) {
  int merge_index = -1;
  const auto &guard = node_item.MutexGuard("UpdateInputForMerge");
  if (!AttrUtils::GetInt(node_item.op_desc, ATTR_NAME_MERGE_INPUT_INDEX, merge_index)) {
    GELOGE(FAILED, "[%s] Get attr %s failed", node_item.NodeName().c_str(), ATTR_NAME_MERGE_INPUT_INDEX.c_str());
    return FAILED;
  }

  if (merge_index < 0 || static_cast<size_t>(merge_index) >= input_tensor_desc.size()) {
    GELOGE(FAILED, "[%s] merge index: %d invalid, should in range[0, %zu)",
           node_item.NodeName().c_str(), merge_index, input_tensor_desc.size());
    return FAILED;
  }

  auto dst_tensor_desc = node_item.MutableInputDesc(merge_index);
  GE_CHECK_NOTNULL(dst_tensor_desc);

  int64_t tensor_size = -1;
  auto &tensor_desc = input_tensor_desc[merge_index];
  (void)TensorUtils::GetSize(tensor_desc, tensor_size);

  dst_tensor_desc->SetShape(tensor_desc.MutableShape());
  dst_tensor_desc->SetOriginShape(tensor_desc.GetOriginShape());
  (void)TensorUtils::SetSize(*dst_tensor_desc, tensor_size);
  (void)guard;
  GELOGD("[%s] Update input shape [%u] with shape: [%s] and ori_shape: [%s], tensor size = %ld",
         node_item.NodeName().c_str(), merge_index, dst_tensor_desc->GetShape().ToString().c_str(),
         dst_tensor_desc->GetOriginShape().ToString().c_str(), tensor_size);

  return SUCCESS;
}

Status ShapeInferenceState::AwaitShapesReady(const GraphExecutionContext &context) {
  if (!node_item.is_dynamic) {
    return SUCCESS;
  }
  std::unique_lock<std::mutex> lk(mu_);
  if (node_item.IsMergeOp()) {
    return UpdateInputForMerge(context);
  }

  if (num_pending_shapes_ > 0) {
    GELOGD("[%s] Await pending shape or shape future start.", node_item.NodeName().c_str());
    int try_count = 0;
    bool wait_success = false;
    while (try_count++ < kMaxWaitTimes) {
      if (ready_cv_.wait_for(lk, std::chrono::seconds(kWaitInternal), [&]() { return num_pending_shapes_ == 0; })) {
        GELOGD("[%s] Await pending shape or shape future end.", node_item.NodeName().c_str());
        wait_success = true;
        break;
      }

      if (context.is_eos_) {
        GELOGD("[%s] Await pending shape cancelled due to end of sequence", node_item.NodeName().c_str());
        return END_OF_SEQUENCE;
      }

      if (context.GetStatus() != SUCCESS) {
        GELOGE(FAILED, "[Check][Status][%s] Await pending shape cancelled.", node_item.NodeName().c_str());
        REPORT_CALL_ERROR("E19999", "[%s] Await pending shape cancelled.", node_item.NodeName().c_str());
        break;
      }
    }

    if (!wait_success) {
      GELOGE(FAILED, "[Check][Status][%s] Wait for shape timeout:%d.", node_item.NodeName().c_str(), kWaitInternal);
      REPORT_CALL_ERROR("E19999", "[%s] Wait for shape timeout:%d.", node_item.NodeName().c_str(), kWaitInternal);
      return FAILED;
    }
  }

  {
    const auto &guard = node_item.MutexGuard("AwaitShapesReady");
    for (size_t i = 0; i < input_tensor_desc.size(); ++i) {
      auto dst_tensor_desc = node_item.MutableInputDesc(i);
      if (dst_tensor_desc == nullptr) {
        continue;
      }

      auto &tensor_desc = input_tensor_desc[i];
      int64_t tensor_size = -1;
      (void)TensorUtils::GetSize(tensor_desc, tensor_size);

      dst_tensor_desc->SetShape(tensor_desc.MutableShape());
      dst_tensor_desc->SetOriginShape(tensor_desc.GetOriginShape());
      (void)TensorUtils::SetSize(*dst_tensor_desc, tensor_size);
    }
    (void)guard;
  }

  for (auto &p : shape_futures) {
    auto idx = p.first;
    auto &future = p.second;
    RECORD_SHAPE_INFERENCE_EVENT(&context, node_item.NodeName().c_str(), "[AwaitShape] [idx = %u] Start", idx);
    const GeTensorDesc* src_tensor_desc = nullptr;
    GE_CHK_STATUS_RET_NOLOG(future.GetTensorDesc(&src_tensor_desc));
    GE_CHECK_NOTNULL(src_tensor_desc);
    RECORD_SHAPE_INFERENCE_EVENT(&context, node_item.NodeName().c_str(), "[AwaitShape] [idx = %u] End", idx);

    int64_t tensor_size = -1;
    (void) TensorUtils::GetSize(*src_tensor_desc, tensor_size);
    GELOGD("[%s] Update input shape [%u] with shape: [%s] and ori_shape: [%s], tensor size = %ld",
           node_item.NodeName().c_str(),
           idx,
           src_tensor_desc->GetShape().ToString().c_str(),
           src_tensor_desc->GetOriginShape().ToString().c_str(),
           tensor_size);
    const auto &guard = node_item.MutexGuard("AwaitShapesReady");
    auto input_desc = node_item.MutableInputDesc(idx);
    GE_CHECK_NOTNULL(input_desc);
    input_desc->SetShape(src_tensor_desc->GetShape());
    input_desc->SetOriginShape(src_tensor_desc->GetOriginShape());
    (void) TensorUtils::SetSize(*input_desc, tensor_size);
    (void)guard;
  }

  return SUCCESS;
}

const vector<GeTensorDesc> &ShapeInferenceState::GetOutputTensorDesc() const {
    return output_tensor_desc;
}

Status ShapeInferenceState::UpdateOutputDesc() {
  for (size_t i = 0; i < output_tensor_desc.size(); ++i) {
    auto src_tensor_desc = node_item.MutableOutputDesc(i);
    GE_CHECK_NOTNULL(src_tensor_desc);
    auto &dst_tensor_desc = output_tensor_desc[i];
    dst_tensor_desc.SetShape(src_tensor_desc->MutableShape());
    dst_tensor_desc.SetOriginShape(src_tensor_desc->GetOriginShape());
    int64_t tensor_size = -1;
    (void) TensorUtils::GetSize(*src_tensor_desc, tensor_size);
    (void) TensorUtils::SetSize(dst_tensor_desc, tensor_size);
  }
  return SUCCESS;
}

ShapeFuture::ShapeFuture(NodeState *src_node,
                         uint32_t src_index,
                         SubgraphContext *subgraph_context)
    : src_node_(src_node), src_index_(src_index), subgraph_context_(subgraph_context) {
}

NodeState::NodeState(const NodeItem &node_item, SubgraphContext *subgraph_context)
    : node_item_(&node_item), shape_inference_state_(node_item), subgraph_context_(subgraph_context) {
  this->op_desc_ = node_item.node->GetOpDesc();
}

Status NodeState::Init(int group, const shared_ptr<FrameState> &frame_state) {
  GE_CHECK_NOTNULL(frame_state);
  group_ = group;
  frame_state_ = frame_state;
  auto unique_task_context = TaskContext::Create(this, subgraph_context_);
  GE_CHECK_NOTNULL(unique_task_context);
  task_context_ = std::shared_ptr<TaskContext>(unique_task_context.release());
  return SUCCESS;
}

Status NodeState::AwaitInputTensors(GraphExecutionContext &context) const {
  if (node_item_->IsMergeOp()) {
    GELOGD("[%s] merge index %d, input nodes: %zu", GetName().c_str(), merge_index_, node_item_->data_recv_.size());
    return SUCCESS;
  }

  for (auto &src_node : node_item_->dependents_for_execution) {
    GELOGD("[%s] Start to wait for data dependent node: [%s]",
           node_item_->NodeName().c_str(),
           src_node->GetName().c_str());
    RECORD_EXECUTION_EVENT(&context,
                           node_item_->NodeName().c_str(),
                           "[AwaitNodeDone] [%s] Start",
                           src_node->GetName().c_str());

    HYBRID_CHK_STATUS_RET(subgraph_context_->Await(src_node),
                          "[%s] Await node [%s] failed.",
                          GetName().c_str(),
                          src_node->GetName().c_str());

    RECORD_EXECUTION_EVENT(&context,
                           node_item_->NodeName().c_str(),
                           "[AwaitNodeDone] [%s] End",
                           src_node->GetName().c_str());
    GELOGD("[%s] Done waiting node: [%s]", node_item_->NodeName().c_str(), src_node->GetName().c_str());
  }

  return SUCCESS;
}

Status NodeState::WaitForPrepareDone() {
  if (prepare_future_.valid()) {
    GELOGD("[%s] Start to wait for prepare future.", GetName().c_str());
    GE_CHK_STATUS_RET(prepare_future_.get(), "[Check][Status][%s] PreRun failed.", GetName().c_str());
  }

  return SUCCESS;
}
Status NodeState::UpdateOutputShapes(int index, const GeShape &shape, const GeShape &ori_shape) {
  auto self_tensor_desc = op_desc_->MutableOutputDesc(index);
  GE_CHECK_NOTNULL(self_tensor_desc);
  self_tensor_desc->SetShape(shape);
  self_tensor_desc->SetOriginShape(ori_shape);
  return SUCCESS;
}

void NodeState::SetTaskContext(std::shared_ptr<TaskContext> &task_context) {
  task_context_ = task_context;
}

std::shared_ptr<TaskContext> NodeState::GetTaskContext() {
  return task_context_;
}

void NodeState::SavePersistTensor(int input_idx, const TensorValue &tensor) {
  const auto is_persist_tensor = [](const std::map<const NodeItem *, std::set<int>> &items, int idx) {
    const auto is_exist = [&idx](const std::pair<const NodeItem *, std::set<int>> &items) {
      return items.second.count(idx) > 0;
    };
    return std::any_of(items.begin(), items.end(), is_exist);
  };

  if (root_tensor_values_.count(input_idx) > 0) {
    return;
  }

  if (is_persist_tensor(node_item_->root_data_, input_idx)) {
    GELOGD("[%s] Save Root input tensor: %d", GetName().c_str(), input_idx);
    root_tensor_values_[input_idx] = tensor;
  } else if (is_persist_tensor(node_item_->enter_data_, input_idx)) {
    GELOGD("[%s] Save Enter input tensor: %d", GetName().c_str(), input_idx);
    root_tensor_values_[input_idx] = tensor;
  }
}

void NodeState::UpdatePersistTensor() {
  const auto update_tensor = [&](const std::map<const NodeItem *, std::set<int>> &items) {
    for (const auto &item : items) {
      for (const auto idx : item.second) {
        UpdatePersistTensor(idx);
      }
    }
  };

  if (root_tensor_values_.empty()) {
    return;
  }

  update_tensor(node_item_->root_data_);
  if (iteration_count_ > 0) {
    update_tensor(node_item_->enter_data_);
  }
}

void NodeState::UpdatePersistTensor(int input_idx) {
  const auto it = root_tensor_values_.find(input_idx);
  if (it == root_tensor_values_.end()) {
    GELOGW("[%s] Not found saved tensor: %d", GetName().c_str(), input_idx);
    return;
  }

  auto tensor = task_context_->MutableInput(input_idx);
  if (tensor == nullptr) {
    GELOGW("[%s] Not found input tensor: %d", GetName().c_str(), input_idx);
    return;
  }

  *tensor = it->second;
  GELOGD("[%s] Update input tensor: %d", GetName().c_str(), input_idx);
}

void NodeState::ResetContext(uint64_t iteration) {
  switch_index_ = -1;
  subgraph_context_->ResetContext(node_item_->node);
  auto unique_task_context = TaskContext::Create(this, subgraph_context_);
  GE_CHECK_NOTNULL_JUST_RETURN(unique_task_context);
  task_context_ = std::shared_ptr<TaskContext>(unique_task_context.release());

  data_scheduled_ = static_cast<uint32_t>(node_item_->root_data_.size());
  ctrl_scheduled_ = static_cast<uint32_t>(node_item_->root_ctrl_.size());
  if (iteration > 0) {
    data_scheduled_ += static_cast<uint32_t>(node_item_->enter_data_.size());
    ctrl_scheduled_ += static_cast<uint32_t>(node_item_->enter_ctrl_.size());
  }

  iteration_count_ = iteration;
  GELOGD("[%s] in while loop, current iteration: %lu, data scheduled: %u, ctrl scheduled: %u, merge index: %d",
         GetName().c_str(), iteration_count_, data_scheduled_, ctrl_scheduled_, merge_index_);
}

void NodeState::ScheduleContext(const NodeState &node_state) {
  if (node_state.node_item_->IsEnterOp()) {
    GELOGD("[%s]{active: %lu, iteration: %lu}, frame{active: %lu, iteration: %lu} [%s]{active: %lu, iteration: %lu}",
           GetName().c_str(), active_count_, iteration_count_, frame_state_->active_count_,
           frame_state_->iteration_count_, node_state.GetName().c_str(), node_state.frame_state_->active_count_,
           node_state.frame_state_->iteration_count_);
    if (frame_state_->active_count_ != active_count_) {
      ResetContext(0);
      active_count_ = frame_state_->active_count_;
    }
  } else if (node_state.node_item_->IsExitOp()) {
    GELOGD("[%s]{active: %lu, iteration: %lu} frame{active: %lu, iteration: %lu} "
           "[%s]{active: %lu, iteration: %lu} parent{active: %lu, iteration: %lu}",
           GetName().c_str(), active_count_, iteration_count_, frame_state_->active_count_,
           frame_state_->iteration_count_, node_state.GetName().c_str(), node_state.frame_state_->active_count_,
           node_state.frame_state_->iteration_count_, node_state.frame_state_->parent_frame_->active_count_,
           node_state.frame_state_->parent_frame_->iteration_count_);
    if (node_state.frame_state_->parent_frame_->iteration_count_ != iteration_count_) {
      ResetContext(node_state.frame_state_->parent_frame_->iteration_count_);
    }
  } else if (node_state.iteration_count_ != iteration_count_) {
    ResetContext(node_state.iteration_count_);
  }
}

Status NodeState::NodeScheduled(const std::function<void(const NodeItem *)> &ready) const {
  // Schedule data output.
  for (const auto &node : node_item_->data_send_) {
    const auto &dst_node_state = subgraph_context_->GetOrCreateNodeState(node);
    GE_CHECK_NOTNULL(dst_node_state);
    dst_node_state->SetDataSchedule(*this, ready);
  }

  // Schedule ctrl output.
  for (const auto &node : node_item_->ctrl_send_) {
    const auto &dst_node_state = subgraph_context_->GetOrCreateNodeState(node);
    GE_CHECK_NOTNULL(dst_node_state);
    dst_node_state->SetCtrlSchedule(*this, ready);
  }

  // Schedule switch group.
  if (switch_index_ >= 0 && static_cast<uint32_t>(switch_index_) < node_item_->switch_groups_.size()) {
    GELOGI("After [%s] scheduled, switch index: %d", GetName().c_str(), switch_index_);
    for (const auto &node : node_item_->switch_groups_[switch_index_]) {
      const auto &dst_node_state = subgraph_context_->GetOrCreateNodeState(node);
      GE_CHECK_NOTNULL(dst_node_state);
      dst_node_state->SetCtrlSchedule(*this, ready);
    }
  }

  return SUCCESS;
}

bool NodeState::IsScheduleReady() const {
  GELOGD("[%s] iteration[%lu] data[input: %zu, scheduled: %u], ctrl[input: %zu+%zu, scheduled: %u]",
         GetName().c_str(), iteration_count_, node_item_->data_recv_.size(), data_scheduled_,
         node_item_->ctrl_recv_.size(), node_item_->GetMergeCtrl(iteration_count_ == 0 ? 0 : 1), ctrl_scheduled_);
  if (node_item_->IsMergeOp()) {
    if (ctrl_scheduled_ != node_item_->GetMergeCtrl(iteration_count_ == 0 ? 0 : 1) + node_item_->ctrl_recv_.size()) {
      return false;
    }

    return data_scheduled_ > 0;
  }

  if (ctrl_scheduled_ != node_item_->ctrl_recv_.size()) {
    return false;
  }

  // Exit may feed loop times...
  return data_scheduled_ >= node_item_->data_recv_.size();
}

void NodeState::SetDataSchedule(const NodeState &node_state, const std::function<void(const NodeItem *)> &ready) {
  GELOGD("[%s] schedule [%s], iteration[%lu -> %lu], data[num: %zu, scheduled: %u], ctrl[num: %zu+%zu, scheduled: %u]",
         node_state.GetName().c_str(), GetName().c_str(), iteration_count_, node_state.iteration_count_,
         node_item_->data_recv_.size(), data_scheduled_, node_item_->ctrl_recv_.size(),
         node_item_->GetMergeCtrl(iteration_count_ == 0 ? 0 : 1), ctrl_scheduled_);

  std::lock_guard<std::mutex> lk(mu_);
  ScheduleContext(node_state);
  ++data_scheduled_;

  if (node_item_->IsMergeOp()) {
    const auto it = node_item_->data_recv_.find(node_state.node_item_);
    if (it != node_item_->data_recv_.end()) {
      merge_index_ = it->second;
      (void)AttrUtils::SetInt(node_item_->node->GetOpDesc(), ATTR_NAME_MERGE_INPUT_INDEX, it->second);
      GELOGD("[%s] scheduled, [%s] set merge index: %d", node_state.GetName().c_str(), GetName().c_str(), it->second);
    } else {
      GELOGW("[%s] scheduled, [%s] not followed", node_state.GetName().c_str(), GetName().c_str());
    }
  }

  if (IsScheduleReady()) {
    ready(node_item_);
  }
}

void NodeState::SetCtrlSchedule(const NodeState &node_state, const std::function<void(const NodeItem *)> &ready) {
  GELOGD("[%s] schedule [%s], iteration[%lu -> %lu], data[num: %zu, scheduled: %u], ctrl[num: %zu+%zu, scheduled: %u]",
         node_state.GetName().c_str(), GetName().c_str(), iteration_count_, node_state.iteration_count_,
         node_item_->data_recv_.size(), data_scheduled_, node_item_->ctrl_recv_.size(),
         node_item_->GetMergeCtrl(iteration_count_ == 0 ? 0 : 1), ctrl_scheduled_);

  std::lock_guard<std::mutex> lk(mu_);
  ScheduleContext(node_state);
  ++ctrl_scheduled_;

  if (IsScheduleReady()) {
    ready(node_item_);
  }
}

void NodeState::RunNextIteration() {
  std::lock_guard<std::mutex> lk(mu_);
  INC_ITERATION_COUNT(iteration_count_);
  ResetContext(iteration_count_);
}

void NodeState::RunStreamActive() {
  std::lock_guard<std::mutex> lk(mu_);
  if (node_item_->ctrl_send_.empty()) {   // Not for Loop Enter or Loop Next.
    return;
  }
  switch_index_ = 0;
  data_scheduled_ = 0;
  ctrl_scheduled_ = 0;
  if (node_item_->is_enter_active_) {
    frame_state_->iteration_count_ = 0;
    INC_ITERATION_COUNT(frame_state_->active_count_);
  } else {
    INC_ITERATION_COUNT(frame_state_->iteration_count_);
  }
  GELOGD("Node[%s] current iteration: %lu, frame active: %lu, frame iteration: %lu",
         GetName().c_str(), iteration_count_, frame_state_->active_count_, frame_state_->iteration_count_);
}

void NodeState::SetScheduleFuture(std::future<Status> &&future) {
  schedule_future_ = std::move(future);
}

Status NodeState::WaitForScheduleDone() {
  if (schedule_future_.valid()) {
    GELOGD("[%s] Start to wait for schedule future.", GetName().c_str());
    GE_CHK_STATUS_RET(schedule_future_.get(), "[Check][Status][%s] wait thread failed", GetName().c_str());
  }

  return SUCCESS;
}

Status ShapeFuture::Get(GeShape &ori_shape, GeShape &shape) {
  GELOGD("Start to wait node: %s for getting shape", src_node_->GetName().c_str());
  HYBRID_CHK_STATUS_RET(subgraph_context_->Await(src_node_->GetNodeItem()->node), "cancelled");
  auto &output_desc = src_node_->GetShapeInferenceState().GetOutputTensorDesc().at(src_index_);
  shape = output_desc.GetShape();
  ori_shape = output_desc.GetOriginShape();
  GELOGD("Get shape from %s:%u. shape = [%s]", src_node_->GetName().c_str(), src_index_, shape.ToString().c_str());
  return SUCCESS;
}

Status ShapeFuture::GetTensorDesc(const GeTensorDesc **tensor_desc) {
  GE_CHECK_NOTNULL(tensor_desc);
  GELOGD("Start to wait node: %s for getting shape", src_node_->GetName().c_str());
  HYBRID_CHK_STATUS_RET(subgraph_context_->Await(src_node_->GetNodeItem()->node), "cancelled");
  *tensor_desc = &src_node_->GetShapeInferenceState().GetOutputTensorDesc().at(src_index_);
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
