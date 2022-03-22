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

#include "graph/manager/graph_manager_utils.h"

#include <set>
#include <utility>

#include "framework/common/debug/ge_log.h"
#include "common/ge/ge_util.h"
#include "framework/common/string_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/optimize/common/params.h"
#include "framework/omg/omg_inner_types.h"
#include "runtime/mem.h"

namespace ge {
using OpDescPtr = std::shared_ptr<OpDesc>;

GraphNode::GraphNode(GraphId graph_id)
    : graph_id_(graph_id),
      run_flag_(false),
      subgraph_ptr_list_(),
      graph_(nullptr),
      compute_graph_(nullptr),
      build_flag_(false),
      load_flag_(false),
      async_(false),
      is_specific_stream_(false),
      ge_model_(nullptr),
      sem_(1) {
  graph_run_async_listener_ = MakeShared<RunAsyncListener>();
  if (graph_run_async_listener_ == nullptr) {
    GELOGE(MEMALLOC_FAILED, "[New][RunAsyncListener] failed");
  }
}

GraphNode::~GraphNode() = default;

void GraphNode::Lock() {
  sem_.Push(0);
}

void GraphNode::Unlock() {
  uint8_t unused;
  sem_.Pop(unused);
}

void GraphNode::IncreaseLoadCount() {
  std::unique_lock<std::mutex> lock(load_count_mu_);
  if (load_record_ == kMaxLoadNum) {
    GELOGW("Reach the maximum of load_count:%u", kMaxLoadNum);
    return;
  }
  ++load_count_;
}

SubGraphInfo::SubGraphInfo() : subgraph_ptr_(nullptr), ge_model_ptr_(nullptr) {}

SubGraphInfo::~SubGraphInfo() {
}

GraphModelListener::GraphModelListener(std::mutex &mutex, std::condition_variable &cond)
    : result_code_(0), is_finished_(false), mutex_(mutex), condition_(cond) {}

Status GraphModelListener::OnComputeDone(uint32_t model_id, uint32_t task_id, uint32_t result,
                                         std::vector<ge::Tensor> &outputs) {
  GELOGI(
      "[GraphManager] graph compute call back, model_id:%u, task_id:%u, "
      "resultCode:%u.",
      model_id, task_id, result);

  std::lock_guard<std::mutex> lock(mutex_);
  result_code_ = result;
  is_finished_ = true;
  condition_.notify_all();

  return SUCCESS;
}

uint32_t GraphModelListener::GetResultCode() const {
  if (!is_finished_) {
    REPORT_CALL_ERROR("E19999", "Model not run finish");
    GELOGE(INTERNAL_ERROR, "[Check][Param] model not run finish.");
    return INTERNAL_ERROR;
  }
  return result_code_;
}

Status GraphModelListener::ResetResult() {
  std::lock_guard<std::mutex> lock(mutex_);
  result_code_ = 0;
  is_finished_ = false;

  return SUCCESS;
}

void RunAsyncListener::SetCallback(const RunAsyncCallback &callback) {
  sem_.Push(0);
  callback_ = callback;
}

Status RunAsyncListener::OnComputeDone(uint32_t model_id, uint32_t task_id, uint32_t result,
                                       std::vector<ge::Tensor> &outputs) {
  GELOGI("[GraphManager] run graph async call back, modelId:%u, taskId:%u, resultCode:%u.",
         model_id, task_id, result);
  GE_CHECK_NOTNULL(callback_);
  callback_(result, outputs);
  uint8_t unused;
  sem_.Pop(unused);
  return SUCCESS;
}

bool HasCalcOp(const ComputeGraphPtr &graph) {
  if (graph == nullptr) {
    return false;
  }

  static const std::set<std::string> calc_op_type = {CONVOLUTION, DECONVOLUTION, FULL_CONNECTION};

  for (const auto &node : graph->GetAllNodes()) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(op_desc == nullptr,
                    REPORT_INNER_ERROR("E19999", "GetOpDesc failed, Node GetOpDesc is nullptr");
                    GELOGE(FAILED, "[Get][OpDesc] failed, Node GetOpDesc is nullptr"); return false);
    if (calc_op_type.find(op_desc->GetType()) != calc_op_type.end()) {
      return true;
    }
  }

  return false;
}
}  // namespace ge
