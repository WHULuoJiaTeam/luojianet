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

#include "graph/passes/variable_ref_delete_op_pass.h"
#include <string>

namespace ge {
Status VariableRefDeleteOpPass::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  std::set<std::string> all_var_names;
  auto root_graph = GraphUtils::FindRootGraph(graph);
  GE_CHECK_NOTNULL(root_graph);
  for (const auto &n : root_graph->GetAllNodes()) {
    all_var_names.insert(n->GetName());
  }
  for (auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    std::string ref_var_src_var_name;
    bool is_variable_ref = (node->GetOpDesc()->GetType() == VARIABLE) &&
                           (ge::AttrUtils::GetStr(node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, ref_var_src_var_name));
    if (!is_variable_ref) {
      continue;
    }
    if (all_var_names.count(ref_var_src_var_name) == 0) {
      REPORT_INNER_ERROR("E19999", "Can not find source variable[%s] of variable ref[%s], check invalid",
                         ref_var_src_var_name.c_str(), node->GetName().c_str());
      GELOGE(FAILED, "[Check][Param] Can not find source variable[%s] of variable ref[%s]",
             ref_var_src_var_name.c_str(), node->GetName().c_str());
      return FAILED;
    }
    Status ret = DealVariableRef(graph, node, ref_var_src_var_name);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Deal][VariableRef] [%s] in graph:%s failed", node->GetName().c_str(), graph->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status VariableRefDeleteOpPass::DealVariableRef(ge::ComputeGraphPtr &graph, ge::NodePtr &variable_ref,
                                                const std::string &ref_var_src_var_name) {
  GE_CHECK_NOTNULL(variable_ref);
  auto inAnchor0 = variable_ref->GetInDataAnchor(0);
  if (inAnchor0 == nullptr) {
    REPORT_INNER_ERROR("E19999", "Node:%s(%s) has no input anchor, check invalid",
                       variable_ref->GetName().c_str(), variable_ref->GetType().c_str());
    GELOGE(FAILED, "[Get][InDataAnchor] failed, variable_ref [%s] no input", variable_ref->GetName().c_str());
    return FAILED;
  }
  GE_CHECK_NOTNULL(inAnchor0->GetPeerOutAnchor());
  // get the output index of the previous node connected to the variable_ref
  // prepare for refreshing address in build phase
  int index = inAnchor0->GetPeerOutAnchor()->GetIdx();

  // get previous node of variable_ref
  NodePtr peer_node = inAnchor0->GetPeerOutAnchor()->GetOwnerNode();

  // add attr [REF_VAR_SRC_VAR_NAME] to the previous op output desc of the variable_ref
  auto op_desc = peer_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  auto out_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(index));
  bool is_set_str = ge::AttrUtils::SetStr(out_desc, REF_VAR_SRC_VAR_NAME, ref_var_src_var_name);
  if (is_set_str) {
    GELOGI("[%s-%d]: add attr [REF_VAR_SRC_VAR_NAME: %s ] ", peer_node->GetName().c_str(), index,
           ref_var_src_var_name.c_str());
  } else {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s to output:%d desc of op:%s(%s) failed", REF_VAR_SRC_VAR_NAME.c_str(),
                      index, op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Set][Attr] %s to output:%d desc of op:%s(%s) failed", REF_VAR_SRC_VAR_NAME.c_str(),
           index, op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }
  // remove variable_ref
  if (GraphUtils::IsolateNode(variable_ref, {0}) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Isolate node:%s(%s) failed",
                      variable_ref->GetName().c_str(), variable_ref->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Isolate][Node] name:%s, type:%s failed", variable_ref->GetName().c_str(),
           variable_ref->GetType().c_str());
    return FAILED;
  }
  if (GraphUtils::RemoveNodeWithoutRelink(graph, variable_ref) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Remove node:%s(%s) without relink in graph:%s failed",
                      variable_ref->GetName().c_str(), variable_ref->GetType().c_str(), graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Remove][Node] %s(%s) without relink in graph:%s failed",
           variable_ref->GetName().c_str(), variable_ref->GetType().c_str(), graph->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace ge
