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
#include "graph/label/case_label_maker.h"

#include "framework/common/util.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "framework/common/op/ge_op_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"

namespace ge {
constexpr uint32_t kCasePredIndex = 0;
constexpr uint32_t kMinCaseBranch = 1;
constexpr uint32_t kMaxCaseBranch = 0x7fffffff;

/**
 * @ingroup ge
 * @brief Make label node to functional call.
 * @param [in/out] label_index: serial id for whole graph.
 * @return: 0 for success / others for fail
 */
Status CaseOpLabelMaker::Run(uint32_t &label_index) {
  GE_CHECK_NOTNULL(parent_node_);
  GE_CHECK_NOTNULL(parent_graph_);

  OpDescPtr case_desc = parent_node_->GetOpDesc();
  GE_CHECK_NOTNULL(case_desc);

  const auto graph_names = case_desc->GetSubgraphInstanceNames();
  if (graph_names.empty() || graph_names.size() > kMaxCaseBranch) {
    REPORT_INNER_ERROR("E19999", "Node:%s(%s) subgraph size: %zu, check invalid", case_desc->GetName().c_str(),
                       case_desc->GetType().c_str(), graph_names.size());
    GELOGE(INTERNAL_ERROR, "[Check][Param] Node: %s has invalid subgraph, graph size: %zu.",
           case_desc->GetName().c_str(), graph_names.size());
    return FAILED;
  }

  // One branch, no need label.
  const uint32_t graph_num = static_cast<uint32_t>(graph_names.size());
  if (graph_num == kMinCaseBranch) {
    GELOGI("Node: %s just one subgraph.", case_desc->GetName().c_str());
    return SUCCESS;
  }

  NodePtr first_label = nullptr;
  ComputeGraphPtr first_graph = nullptr;
  std::vector<uint32_t> switch_labels;
  uint32_t last_label_index = label_index++;
  for (uint32_t index = 0; index < graph_num; ++index) {
    ComputeGraphPtr graph = parent_graph_->GetSubgraph(graph_names[index]);
    GE_CHECK_NOTNULL(graph);

    // all branch, add label and stream active nodes to head.
    std::string stream_active_name =
      parent_node_->GetName() + "/StreamActive_" + std::to_string(index);  // rtStreamActive
    NodePtr stream_active = AddStreamActive(graph, stream_active_name);
    if (stream_active == nullptr) {
      REPORT_CALL_ERROR("E19999", "Add StreamActive node in graph:%s fail",
                        graph->GetName().c_str());
      GELOGE(INTERNAL_ERROR, "[Add][StreamActive] in Subgraph: %s failed.", graph->GetName().c_str());
      return FAILED;
    }

    uint32_t curr_label_index = label_index++;
    std::string label_set_name = parent_node_->GetName() + "/LabelSet_" + std::to_string(index);  // rtLabelSet
    NodePtr label = AddLabelSetEnter(graph, label_set_name, curr_label_index, stream_active);
    if (label == nullptr) {
      REPORT_CALL_ERROR("E19999", "Add LabelSetEnter node in graph:%s fail",
                        graph->GetName().c_str());
      GELOGE(INTERNAL_ERROR, "[Call][AddLabelSetEnter] Subgraph: %s add label set failed.", graph->GetName().c_str());
      return FAILED;
    }
    switch_labels.emplace_back(curr_label_index);
    if (index == 0) {  // save first subgraph node for switch.
      first_label = label;
      first_graph = graph;
    }

    if (index + 1 < graph_num) {
      // middle node, add goto node to tail.
      std::string label_goto_name = parent_node_->GetName() + "/LabelGoto_" + std::to_string(index);  // rtLabelGoto
      if (AddLabelGotoLeave(graph, label_goto_name, last_label_index) == nullptr) {
        REPORT_CALL_ERROR("E19999", "Add LabelGotoLeave node in graph:%s fail",
                          graph->GetName().c_str());
        GELOGE(INTERNAL_ERROR, "[Call][AddLabelGotoLeave] Subgraph: %s add label goto failed.",
               graph->GetName().c_str());
        return FAILED;
      }
    } else {
      // last node, add label node to tail.
      std::string last_label_name = parent_node_->GetName() + "/LabelSet_Last";  // rtLabelSet
      if (AddLabelSetLeave(graph, last_label_name, last_label_index) == nullptr) {
        REPORT_CALL_ERROR("E19999", "Add LabelSetLeave node in graph:%s fail",
                          graph->GetName().c_str());
        GELOGE(INTERNAL_ERROR, "[Call][AddLabelSetLeave] Subgraph: %s add label set failed.",
               graph->GetName().c_str());
        return FAILED;
      }
    }
  }

  // Add Switch node for first branch.
  GE_CHECK_NOTNULL(first_label);
  GE_CHECK_NOTNULL(first_graph);

  // first case, add switch node to head.
  const std::string label_switch_name = parent_node_->GetName() + "/LabelSwitch";  // rtLabelSwitchByIndex
  const GeTensorDesc &pred_desc = case_desc->GetInputDesc(kCasePredIndex);
  NodePtr switch_node = AddLabelSwitchEnter(first_graph, label_switch_name, pred_desc, switch_labels);
  if (switch_node == nullptr) {
    REPORT_CALL_ERROR("E19999", "Add LabelSwitchEnter node in graph:%s fail",
                      first_graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Call][AddLabelSwitchEnter] Subgraph: %s add label switch failed.",
           first_graph->GetName().c_str());
    return FAILED;
  }

  // Link control edge to then branch head.
  if (GraphUtils::AddEdge(switch_node->GetOutControlAnchor(), first_label->GetInControlAnchor()) != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add ctrl edge from %s to %s in graph:%s fail", switch_node->GetName().c_str(),
                      first_label->GetName().c_str(), first_graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][CtrlEdge] to %s failed.", first_label->GetName().c_str());
    return FAILED;
  }

  uint32_t parent_index = 0;  // Case cond input is first.
  const std::string data_name = parent_node_->GetName() + "/SwitchIndexData";
  if (AddLabelSwitchIndex(first_graph, data_name, pred_desc, switch_node, parent_index) == nullptr) {
    REPORT_CALL_ERROR("E19999", "Add LabelSwitchIndex node in graph:%s fail",
                      first_graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Call][AddLabelSwitchIndex] Subgraph: %s add switch input failed.",
           first_graph->GetName().c_str());
    return FAILED;
  }

  GELOGI("Node: %s assign label success.", case_desc->GetName().c_str());
  return SUCCESS;
}

REGISTER_LABEL_MAKER(CASE, CaseOpLabelMaker);
}  // namespace ge
