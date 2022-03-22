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

#include "graph/passes/end_of_sequence_add_control_pass.h"

#include <string>
#include <vector>

#include "init/gelib.h"
#include "graph/node.h"

namespace ge {

Status EndOfSequenceAddControlPass::Run(ComputeGraphPtr graph) {
  if (graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param graph is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Check][Param] param [graph] must not be null.");
    return PARAM_INVALID;
  }
  if (graph->GetParentGraph() != nullptr) {
    return SUCCESS;
  }
  NodePtr end_of_sequence = GetEndOfSequence(graph);
  if (end_of_sequence == nullptr) {
    return SUCCESS;
  }
  GELOGI("EndOfSequenceAddControlPass begin.");

  std::vector<NodePtr> target_nodes;
  for (NodePtr &node : graph->GetDirectNode()) {
    if (node == nullptr) {
      GELOGW("node is nullptr.");
      continue;
    }
    string stream_label;
    (void)AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label);
    if (!stream_label.empty() || IsDataLikeNode(node)) {
      continue;
    }
    // Save the nodes whose pre-nodes are all data-like node
    auto in_data_nodes = node->GetInDataNodes();
    bool flag = false;
    for (auto in_node : in_data_nodes) {
      if (!IsDataLikeNode(in_node)) {
        flag = true;
        break;
      }
    }
    if (flag) {
      continue;
    }
    target_nodes.push_back(node);
  }
  // Insert control edge
  Status status = AddControlEdge(end_of_sequence, target_nodes);
  if (status != SUCCESS) {
    GELOGE(FAILED, "[Add][ControlEdge] Graph add EndOfSequence op:%s out ctrl edge failed.",
           end_of_sequence->GetName().c_str());
    return FAILED;
  }
  GELOGI("EndOfSequenceAddControlPass end.");
  return SUCCESS;
}

Status EndOfSequenceAddControlPass::AddControlEdge(NodePtr &end_of_sequence, std::vector<NodePtr> &target_nodes) {
  auto out_ctrl_anchor = end_of_sequence->GetOutControlAnchor();
  for (NodePtr &node : target_nodes) {
    auto in_ctrl_anchor = node->GetInControlAnchor();
    if (in_ctrl_anchor == nullptr) {
      continue;
    }
    Status status = GraphUtils::AddEdge(out_ctrl_anchor, in_ctrl_anchor);
    if (status != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                        end_of_sequence->GetName().c_str(), end_of_sequence->GetType().c_str(),
                        node->GetName().c_str(), node->GetType().c_str());
      GELOGE(FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
             end_of_sequence->GetName().c_str(), end_of_sequence->GetType().c_str(),
             node->GetName().c_str(), node->GetType().c_str());
      return FAILED;
    }
    GELOGI("Graph add EndOfSequence op out ctrl edge, dst node: %s.", node->GetName().c_str());
  }
  return SUCCESS;
}

inline NodePtr EndOfSequenceAddControlPass::GetEndOfSequence(const ComputeGraphPtr &graph) const {
  // Internal function, guaranteeing graph non-null
  for (NodePtr &node : graph->GetDirectNode()) {
    if (node->GetType() == ENDOFSEQUENCE) {
      return node;
    }
  }
  return nullptr;
}

bool EndOfSequenceAddControlPass::IsDataLikeNode(const NodePtr &node) {
  std::shared_ptr<GELib> instance_ptr = GELib::GetInstance();
  if ((instance_ptr == nullptr) || (!instance_ptr->InitFlag())) {
    GELOGW("GELib not initialized");
    return false;
  }
  OpDescPtr op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    return false;
  }
  string engine_name = op_desc->GetOpEngineName();
  if (engine_name.empty()) {
    engine_name = instance_ptr->DNNEngineManagerObj().GetDNNEngineName(node);
  }
  const map<string, SchedulerConf> schedulers = instance_ptr->DNNEngineManagerObj().GetSchedulers();
  // Only one scheduler has been supported by now
  for (auto schedulers_iter = schedulers.begin(); schedulers_iter != schedulers.end(); ++schedulers_iter) {
    const map<string, EngineConfPtr> cal_engines = schedulers_iter->second.cal_engines;
    auto cal_engines_iter = cal_engines.find(engine_name);
    if (cal_engines_iter == cal_engines.end()) {
      GELOGW("No cal_engines found within engine %s, node name %s", engine_name.c_str(), node->GetName().c_str());
      continue;
    }
    EngineConfPtr engine_conf_ptr = cal_engines_iter->second;
    if (engine_conf_ptr == nullptr) {
      GELOGW("engine_conf_ptr within engine %s, node name %s is null", engine_name.c_str(), node->GetName().c_str());
      continue;
    }
    bool skip_assign_stream = engine_conf_ptr->skip_assign_stream;
    if (skip_assign_stream) {
      return true;
    }
    return false;
  }
  return false;
}
}  // namespace ge
