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

#include "graph/manager/util/variable_accelerate_ctrl.h"

#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"

namespace ge {
namespace {
inline bool IsVariable(const std::string &node_type) {
  return node_type == VARIABLE || node_type == VARIABLEV2 || node_type == VARHANDLEOP;
}
}

bool VarAccelerateCtrl::IsVarPermitToChangeFormats(const std::string &var_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = var_names_to_change_times_.find(var_name);
  if (iter == var_names_to_change_times_.end()) {
    return true;
  }
  return iter->second < kMaxVarChangeTimes_;
}

void VarAccelerateCtrl::SetVarChanged(const std::string &var_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto times = ++var_names_to_change_times_[var_name];
  for (auto &graph_id_to_var_names : graph_ids_to_var_names_) {
    if (graph_id_to_var_names.second.count(var_name) > 0) {
      GELOGI("The format of var %s has been changed, total changed times %d, "
             "the graph %u contains which should be re-build before next run",
             var_name.c_str(), times, graph_id_to_var_names.first);
      /// The graph being compiled right now is also added to the rebuild-list
      /// and can be deleted by calling `SetGraphBuildEnd` at the end of compilation.
      graph_ids_need_rebuild_.insert(graph_id_to_var_names.first);
    }
  }
}

void VarAccelerateCtrl::AddGraph(uint32_t graph_id, const ComputeGraphPtr &compute_graph) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (compute_graph == nullptr) {
    GELOGE(PARAM_INVALID, "[Check][Param] Failed to add graph %u, the compute graph is null", graph_id);
    return;
  }
  auto &var_names = graph_ids_to_var_names_[graph_id];
  for (auto &node : compute_graph->GetAllNodes()) {
    auto node_type = node->GetType();
    if (IsVariable(node_type)) {
      GELOGD("Add graph %u contains variable %s", graph_id, node->GetName().c_str());
      var_names.insert(node->GetName());
    }
  }
  GELOGD("Add graph %u, var count %zu", graph_id, var_names.size());
}

void VarAccelerateCtrl::RemoveGraph(uint32_t graph_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  GELOGD("Remove graph %u", graph_id);
  graph_ids_to_var_names_.erase(graph_id);
  graph_ids_need_rebuild_.erase(graph_id);
}

bool VarAccelerateCtrl::IsGraphNeedRebuild(uint32_t graph_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return graph_ids_need_rebuild_.count(graph_id) > 0;
}

void VarAccelerateCtrl::SetGraphBuildEnd(uint32_t graph_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  graph_ids_need_rebuild_.erase(graph_id);
  GELOGD("The graph %u has built end, remove it from the rebuild-set", graph_id);
}
}  // namespace ge
