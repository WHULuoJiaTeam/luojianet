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

#include "graph/partition/engine_place.h"

#include <climits>
#include <memory>
#include <string>
#include <utility>
#include <mutex>

#include "framework/common/op/ge_op_utils.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "init/gelib.h"
#include "opskernel_manager/ops_kernel_manager.h"
#include "analyzer/analyzer.h"

namespace ge {
namespace {
std::mutex check_support_cost_mutex;
}
Status EnginePlacer::Check() const {
  if (compute_graph_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "compute_graph_ is nullptr, check invalid.");
    GELOGE(GE_GRAPH_NULL_INPUT, "[Check][Param] compute_graph_ is nullptr.");
    return FAILED;
  }
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if ((instance_ptr == nullptr) || (!instance_ptr->InitFlag())) {
    REPORT_INNER_ERROR("E19999", "GELib instance is nullptr or it is not InitFlag, check invalid.");
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Get][GELib] Run enginePlacer failed, because GELib is invalid.");
    return FAILED;
  }
  return SUCCESS;
}

Status EnginePlacer::Run() {
  std::lock_guard<std::mutex> lock(check_support_cost_mutex);

  GELOGD("Engine placer starts.");
  if (Check() != SUCCESS) {
    return FAILED;
  }
  bool is_check_support_success = true;
  // Assign engine for each node in the graph
  ge::GELib::GetInstance()->DNNEngineManagerObj().InitPerformanceStaistic();
  for (const auto &node_ptr : compute_graph_->GetDirectNode()) {
    GE_CHECK_NOTNULL(node_ptr);
    auto op_desc = node_ptr->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    std::string engine_name;
    std::string kernel_name;
    // Check if this node has assigned engine
    bool has_engine_attr =
        AttrUtils::GetStr(op_desc, ATTR_NAME_ENGINE_NAME_FOR_LX, engine_name) && !engine_name.empty();
    bool has_kernel_attr =
        AttrUtils::GetStr(op_desc, ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX, kernel_name) && !kernel_name.empty();
    bool use_exist_engine_name = !op_desc->GetOpKernelLibName().empty() || (has_kernel_attr && has_engine_attr);
    if (use_exist_engine_name) {
      if (op_desc->GetOpEngineName().empty()) {
        GELOGI("Op %s set engine_name %s engine_name %s from attrs",
               op_desc->GetName().c_str(),
               engine_name.c_str(),
               kernel_name.c_str());
        op_desc->SetOpEngineName(engine_name);
        op_desc->SetOpKernelLibName(kernel_name);
      }
      engine_name = op_desc->GetOpEngineName();
    } else {
      // Call placer cost model to get the "best" engine for this node
      engine_name = ge::GELib::GetInstance()->DNNEngineManagerObj().GetDNNEngineName(node_ptr);
      // If can't get op's engine name, keep check support finish and return failed
      if (engine_name.empty()) {
        is_check_support_success = false;
        ErrorManager::GetInstance().ATCReportErrMessage(
            "E13003", {"opname", "optype"}, {op_desc->GetName(), op_desc->GetType()});
        GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Check][Param] Can not find engine of op name %s type %s",
               op_desc->GetName().c_str(), op_desc->GetType().c_str());
        continue;
      }
    }
    if (AssignEngineAndLog(node_ptr, engine_name) != SUCCESS) {
      GELOGE(GE_GRAPH_ASSIGN_ENGINE_FAILED, "[Call][AssignEngineAndLog] FAILED, node:%s", op_desc->GetName().c_str());
      return FAILED;
    }
  }

  for (auto &it : ge::GELib::GetInstance()->DNNEngineManagerObj().GetCheckSupportCost()) {
    GEEVENT("The time cost of %s::CheckSupported is [%lu] micro second.", it.first.c_str(), it.second);
  }
  GELOGD("Engine placer ends.");
  return is_check_support_success ? SUCCESS : FAILED;
}

Status EnginePlacer::AssignEngineAndLog(ge::ConstNodePtr node_ptr, const std::string &engine_name) {
  if ((node_ptr == nullptr) || (node_ptr->GetOpDesc() == nullptr)) {
    REPORT_INNER_ERROR("E19999", "Param node_ptr is nullptr or it's opdesc is nullptr, check invalid.");
    GELOGE(FAILED, "[Check][Param] node_ptr is nullptr.");
    return FAILED;
  }

  // private function, promise node_ptr->GetOpDesc() not null
  GELOGD("Assigning DNNEngine %s to node %s, op type %s", engine_name.c_str(), node_ptr->GetName().c_str(),
         node_ptr->GetOpDesc()->GetType().c_str());

  // Record the node assigned engine name
  node_engine_map_.insert(std::make_pair(node_ptr, engine_name));

  return SUCCESS;
}
}  // namespace ge

