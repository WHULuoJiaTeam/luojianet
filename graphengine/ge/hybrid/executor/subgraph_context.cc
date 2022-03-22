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

#include "hybrid/executor/subgraph_context.h"
#include "hybrid/executor/hybrid_model_executor.h"

namespace ge {
namespace hybrid {
SubgraphContext::SubgraphContext(const GraphItem *graph_item, GraphExecutionContext *execution_context)
    : graph_item_(graph_item), execution_context_(execution_context) {
}

SubgraphContext::~SubgraphContext() {
  if (mmRWLockDestroy(&rw_lock_) != EN_OK) {
    REPORT_CALL_ERROR("E19999", "Destroy rw_lock failed");
    GELOGE(INTERNAL_ERROR, "[RWLock][Destroy] Destroy rw_lock failed");
  }
}

Status SubgraphContext::Init() {
  GE_CHECK_NOTNULL(graph_item_);
  GELOGD("[%s] Start to init subgraph context. total inputs = %d, total outputs = %d",
         graph_item_->GetName().c_str(),
         graph_item_->TotalInputs(),
         graph_item_->TotalOutputs());
  all_inputs_.resize(static_cast<unsigned long>(graph_item_->TotalInputs()));
  all_outputs_.resize(static_cast<unsigned long>(graph_item_->TotalOutputs()));
  if (mmRWLockInit(&rw_lock_) != EN_OK) {
    REPORT_CALL_ERROR("E19999", "Init rw_lock failed");
    GELOGE(INTERNAL_ERROR, "[RWLock][Init] Init rw_lock failed");
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

void SubgraphContext::SetGroup(int group) {
  group_ = group;
}

void SubgraphContext::ResetContext(const NodePtr &node) {
  node_done_manager_.Reset(node);
}

NodeStatePtr SubgraphContext::GetOrCreateNodeState(const NodeItem *node_item) {
  GELOGD("[%s] lock for read", node_item->NodeName().c_str());
  if (mmRWLockRDLock(&rw_lock_) != EN_OK) {
    REPORT_CALL_ERROR("E19999", "[Node:%s] Lock for read failed", node_item->NodeName().c_str());
    GELOGE(INTERNAL_ERROR, "[RWLock][Lock][Node:%s] Lock for read failed", node_item->NodeName().c_str());
    return nullptr;
  }
  const auto &iter = node_states_.find(node_item);
  if (iter != node_states_.end()) {
    auto state = iter->second;
    GELOGD("[%s] unlock for read", node_item->NodeName().c_str());
    if (mmRDLockUnLock(&rw_lock_) != EN_OK) {
      REPORT_CALL_ERROR("E19999", "[Node:%s] Unlock for read failed", node_item->NodeName().c_str());
      GELOGE(INTERNAL_ERROR, "[RWLock][Unlock][Node:%s] Unlock for read failed", node_item->NodeName().c_str());
      return nullptr;
    }
    return state;
  }
  GELOGD("[%s] unlock for read", node_item->NodeName().c_str());
  if (mmRDLockUnLock(&rw_lock_) != EN_OK) {
    REPORT_CALL_ERROR("E19999", "[Node:%s] Unlock for read failed", node_item->NodeName().c_str());
    GELOGE(INTERNAL_ERROR, "[RWLock][Unlock][Node:%s] Unlock for read failed", node_item->NodeName().c_str());
    return nullptr;
  }

  return CreateNodeState(node_item);
}

NodeStatePtr SubgraphContext::CreateNodeState(const NodeItem *node_item) {
  GELOGD("[%s] lock for write", node_item->NodeName().c_str());
  if (mmRWLockWRLock(&rw_lock_) != EN_OK) {
    REPORT_CALL_ERROR("E19999", "[Node:%s] Lock for write failed", node_item->NodeName().c_str());
    GELOGE(INTERNAL_ERROR, "[RWLock][Lock][Node:%s] Lock for write failed", node_item->NodeName().c_str());
    return nullptr;
  }

  auto &node_state = node_states_[node_item];
  do {
    if (node_state == nullptr) {
      const auto &guard = node_item->MutexGuard("GetOrCreateNodeState");
      node_state.reset(new(std::nothrow)NodeState(*node_item, this));
      if (node_state == nullptr || node_state->Init(group_, GetOrCreateFrameState(*node_item)) != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "[Create][NodeState] failed for[%s].", node_item->NodeName().c_str());
        REPORT_CALL_ERROR("E19999", "Create NodeState failed for %s.", node_item->NodeName().c_str());
        break;
      }
      (void)guard;
    }
  } while (0);

  GELOGD("[%s] unlock for write", node_item->NodeName().c_str());
  if (mmWRLockUnLock(&rw_lock_) != EN_OK) {
    REPORT_CALL_ERROR("E19999", "[Node:%s] Unlock for write failed", node_item->NodeName().c_str());
    GELOGE(INTERNAL_ERROR, "[RWLock][Unlock][Node:%s] Unlock for write failed", node_item->NodeName().c_str());
    return nullptr;
  }

  return node_state;
}

FrameStatePtr SubgraphContext::GetOrCreateFrameState(const NodeItem &node_item) {
  auto &frame_state = frame_states_[node_item.frame_index_];
  if (frame_state == nullptr) {
    GELOGD("[%s] Create FrameState, frame index: %ld, parent frame index: %ld",
           node_item.node_name.c_str(), node_item.frame_index_, node_item.parent_frame_);
    frame_state.reset(new(std::nothrow)FrameState(node_item.frame_index_));
    if (node_item.frame_index_ != -1) {  // -1 is root frame.
      frame_state->parent_frame_ = frame_states_[node_item.parent_frame_];
    }
  }

  return frame_state;
}

Status SubgraphContext::SetInput(int index, const TensorValue &tensor) {
  if (static_cast<size_t>(index) >= all_inputs_.size()) {
    GELOGE(INTERNAL_ERROR,
           "[Check][Param:index]input index out of range. all input num = %zu, input index = %d",
           all_inputs_.size(), index);
    REPORT_INNER_ERROR("E19999", "input param index out of range, all input num = %zu, input index = %d.",
                       all_inputs_.size(), index);
    return INTERNAL_ERROR;
  }
  all_inputs_[index] = tensor;
  return SUCCESS;
}

Status SubgraphContext::SetInput(const NodeItem &node_item, int input_index, const TensorValue &tensor) {
  auto index = node_item.input_start + input_index;
  return SetInput(index, tensor);
}

Status SubgraphContext::SetOutput(const NodeItem &node_item, int output_index, const TensorValue &tensor) {
  auto index = node_item.output_start + output_index;
  if ((output_index >= node_item.num_outputs) || (static_cast<size_t>(index) >= all_outputs_.size())) {
    GELOGE(INTERNAL_ERROR, "[Check][Param:output_index]output index out of range. all output num = %zu,"
           "node_item = %s, output index = %d.",
           all_outputs_.size(), node_item.DebugString().c_str(), output_index);
    REPORT_INNER_ERROR("E19999", "output index out of range. all output num = %zu, node_item = %s, output index = %d.",
                       all_outputs_.size(), node_item.DebugString().c_str(), output_index);
    return INTERNAL_ERROR;
  }

  all_outputs_[index] = tensor;
  return SUCCESS;
}

Status SubgraphContext::GetInput(int index, TensorValue &tensor) {
  GE_CHECK_GE(all_inputs_.size(), index + 1U);
  tensor = all_inputs_[index];
  return SUCCESS;
}

Status SubgraphContext::GetOutputs(std::vector<TensorValue> &outputs) {
  if (graph_item_->IsDynamic()) {
    GELOGD("[%s] graph is dynamic, get outputs from net output input tensors", graph_item_->GetName().c_str());
    // get from net output inputs
    auto output_node = graph_item_->GetOutputNode();
    if (output_node != nullptr) {
      for (int i = 0; i < output_node->num_inputs; ++i) {
        TensorValue tensor;
        GE_CHK_STATUS_RET_NOLOG(GetInput(output_node->input_start + i, tensor));
        GELOGD("[%s] Adding output tensor by input index [%d], tensor = %s",
               graph_item_->GetName().c_str(),
               output_node->input_start + i,
               tensor.DebugString().c_str());
        outputs.emplace_back(std::move(tensor));
      }
    }
  } else {
    GELOGD("[%s] graph is non-dynamic, get outputs from subgraph outputs", graph_item_->GetName().c_str());
    for (auto &tensor : all_outputs_) {
      GELOGD("[%s] Adding output tensor: %s", graph_item_->GetName().c_str(), tensor.DebugString().c_str());
      outputs.emplace_back(tensor);
    }
  }

  return SUCCESS;
}

Status SubgraphContext::Await(const NodePtr &node) {
  if (node_done_manager_.Await(node)) {
    return SUCCESS;
  }

  if (execution_context_->is_eos_) {
    return END_OF_SEQUENCE;
  }

  return FAILED;
}

void SubgraphContext::OnError(Status error) {
  if (error != END_OF_SEQUENCE) {
    GELOGE(error, "[Check][Param:error][%s] Error:%d occurred while executing graph.",
           graph_item_->GetName().c_str(), error);
    REPORT_INNER_ERROR("E19999", "[%s] Error:%d occurred while executing graph.",
                       graph_item_->GetName().c_str(), error);
  }
  node_done_manager_.Destroy();
}

void SubgraphContext::NodeDone(const NodePtr &node) {
  node_done_manager_.NodeDone(node);
}

void SubgraphContext::Reset() {
  node_done_manager_.Reset();
  if (mmRWLockWRLock(&rw_lock_) == EN_OK) {
    node_states_.clear();
    (void)mmWRLockUnLock(&rw_lock_);
  }
}
}  // namespace hybrid
}  // namespace ge
