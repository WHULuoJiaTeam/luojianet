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

#include "graph/passes/switch_to_stream_switch_pass.h"
#include <stack>
#include "common/ge/ge_util.h"
#include "external/ge/ge_api_types.h"
#include "common/omg_util.h"
#include "graph/ge_context.h"
#include "graph/utils/type_utils.h"

namespace ge {
Status SwitchToStreamSwitchPass::Run(ComputeGraphPtr graph) {
  GELOGD("SwitchToStreamSwitchPass Enter");

  GE_CHK_STATUS_RET(CheckCycleDependence(graph),
                    "[Check][CycleDependence] in graph:%s failed.", graph->GetName().c_str());
  for (const auto &switch_node : switch_nodes_) {
    GE_CHK_STATUS_RET(ReplaceSwitchNode(graph, switch_node),
                      "[Replace][Switch] by StreamSwitch in graph:%s failed.", graph->GetName().c_str());
  }
  GE_CHK_STATUS_RET(CombineSwitchNode(graph),
                    "[Combine][SwitchNode] in graph:%s failed.", graph->GetName().c_str());

  for (const auto &node : bypass_nodes_) {
    GE_CHK_BOOL_EXEC(graph->IsolateNode(node) == GRAPH_SUCCESS,
                     REPORT_CALL_ERROR("E19999", "Isolate node:%s(%s) in graph:%s failed",
                                       node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
                     return FAILED,
                     "[Isolate][Node] %s(%s) in graph:%s failed.",
                     node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
    GE_CHK_BOOL_EXEC(GraphUtils::RemoveNodeWithoutRelink(graph, node) == GRAPH_SUCCESS,
                     REPORT_CALL_ERROR("E19999", "Remove node:%s(%s) without relink in graph:%s failed",
                                       node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
                     return FAILED,
                     "[Remove][Node] %s(%s) without relink in graph:%s failed",
                     node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
  }

  GELOGD("SwitchToStreamSwitchPass Leave");
  return SUCCESS;
}

///
/// @brief Clear Status
/// @return
///
Status SwitchToStreamSwitchPass::ClearStatus() {
  switch_nodes_.clear();
  switch_cyclic_map_.clear();
  bypass_nodes_.clear();
  stream_switch_nodes_.clear();
  cond_node_map_.clear();
  switch_node_map_.clear();
  node_num_map_.clear();
  return SUCCESS;
}

///
/// @brief Check cyclic dependence
/// @param [in] graph
/// @return Status
///
Status SwitchToStreamSwitchPass::CheckCycleDependence(const ComputeGraphPtr &graph) {
  std::string type;
  std::unordered_map<NodePtr, std::vector<NodePtr>> cond_switch_map;
  for (const NodePtr &node : graph->GetDirectNode()) {
    GE_CHK_STATUS_RET(GetOriginalType(node, type),
                      "[Get][OriginalType] failed, graph:%s.", graph->GetName().c_str());
    if ((type != SWITCH) && (type != REFSWITCH)) {
      continue;
    }
    InDataAnchorPtr in_cond_anchor = node->GetInDataAnchor(SWITCH_PRED_INPUT);
    GE_CHECK_NOTNULL(in_cond_anchor);
    OutDataAnchorPtr peer_out_anchor = in_cond_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor);
    if (FindSwitchCondInput(peer_out_anchor) != SUCCESS) {
      GELOGE(FAILED, "[Find][PredInput] for switch_node %s failed.", node->GetName().c_str());
      return FAILED;
    }

    NodePtr cond_node = peer_out_anchor->GetOwnerNode();
    auto iter = cond_switch_map.find(cond_node);
    if (iter == cond_switch_map.end()) {
      cond_switch_map[cond_node] = { node };
    } else {
      iter->second.emplace_back(node);
    }
    switch_nodes_.emplace_back(node);
  }

  MarkCycleDependence(cond_switch_map);
  return SUCCESS;
}

///
/// @brief Mark cyclic dependence
/// @param [in] graph
/// @param [in] cond_switch_map
/// @return void
///
void SwitchToStreamSwitchPass::MarkCycleDependence(
    const std::unordered_map<NodePtr, std::vector<NodePtr>> &cond_switch_map) {
  std::stack<NodePtr> out_nodes;
  NodePtr tmp_node = nullptr;
  std::unordered_set<NodePtr> visited;
  for (const auto &iter : cond_switch_map) {
    std::set<NodePtr> switch_nodes(iter.second.begin(), iter.second.end());
    for (const auto &switch_node : switch_nodes) {
      GELOGD("MarkCycleDependence: cond_node=%s, switch=%s.", iter.first->GetName().c_str(),
             switch_node->GetName().c_str());
      for (const auto &node : switch_node->GetOutAllNodes()) {
        out_nodes.push(node);
      }
    }
    visited.clear();
    while (!out_nodes.empty()) {
      tmp_node = out_nodes.top();
      out_nodes.pop();
      if (visited.count(tmp_node) > 0) {
        continue;
      }
      for (const NodePtr &out_node : tmp_node->GetOutAllNodes()) {
        if (switch_nodes.find(out_node) == switch_nodes.end()) {
          out_nodes.push(out_node);
          continue;
        }
        GELOGD("MarkCycleDependence: tmp_node=%s, switch_node=%s.",
               tmp_node->GetName().c_str(), out_node->GetName().c_str());
        GE_IF_BOOL_EXEC(SetCyclicDependenceFlag(out_node) != SUCCESS,
                        GELOGW("set cyclic dependence attr failed."); return );
        auto map_iter = switch_cyclic_map_.find(out_node);
        if (map_iter == switch_cyclic_map_.end()) {
          switch_cyclic_map_[out_node] = {tmp_node->GetName()};
        } else {
          map_iter->second.insert(tmp_node->GetName());
        }
      }
      visited.insert(tmp_node);
    }
  }

  return;
}

///
/// @brief Replace Switch Op
/// @param [in] graph
/// @param [in] switch_node
/// @return Status
///
Status SwitchToStreamSwitchPass::ReplaceSwitchNode(const ComputeGraphPtr &graph, const NodePtr &switch_node) {
  OutDataAnchorPtr peer_data_anchor = nullptr;
  OutDataAnchorPtr peer_cond_anchor = nullptr;
  GE_CHK_BOOL_EXEC(BypassSwitchNode(switch_node, peer_data_anchor, peer_cond_anchor) == SUCCESS, return FAILED,
                   "[Bypass][SwitchNode] %s failed.", switch_node->GetName().c_str());
  GE_CHECK_NOTNULL(peer_data_anchor);
  GE_CHECK_NOTNULL(peer_cond_anchor);
  OpDescPtr cond_desc = peer_cond_anchor->GetOwnerNode()->GetOpDesc();
  GE_CHECK_NOTNULL(cond_desc);
  DataType cond_data_type = cond_desc->GetOutputDesc(peer_cond_anchor->GetIdx()).GetDataType();
  GE_CHK_BOOL_EXEC(cond_data_type == DT_BOOL,
                   REPORT_INNER_ERROR("E19999", "Pred_input of Switch node:%s(%s) only support DT_BOOL data_type, "
                                      "but %s exactly", switch_node->GetName().c_str(), switch_node->GetType().c_str(),
                                      TypeUtils::DataTypeToSerialString(cond_data_type).c_str());
                   return FAILED,
                   "[Check][Param] Pred_input of Switch node:%s(%s) only support DT_BOOL data_type, but %s exactly",
                   switch_node->GetName().c_str(), switch_node->GetType().c_str(),
                   TypeUtils::DataTypeToSerialString(cond_data_type).c_str());

  OpDescPtr switch_desc = switch_node->GetOpDesc();
  GE_CHECK_NOTNULL(switch_desc);
  bool cyclic_flag = switch_desc->HasAttr(ATTR_NAME_CYCLIC_DEPENDENCE_FLAG);
  std::set<std::string> out_node_list;
  for (const auto &out_data_anchor : switch_node->GetAllOutDataAnchors()) {
    bool true_branch_flag = (static_cast<uint32_t>(out_data_anchor->GetIdx()) == SWITCH_TRUE_OUTPUT);
    NodePtr stream_switch = nullptr;
    out_node_list.clear();
    for (const auto &peer_in_anchor : out_data_anchor->GetPeerAnchors()) {
      GE_IF_BOOL_EXEC(stream_switch == nullptr, {
        stream_switch = CreateStreamSwitchNode(graph, switch_node, true_branch_flag ? "_t" : "_f", peer_cond_anchor);
        GE_CHK_BOOL_EXEC(stream_switch != nullptr, return FAILED,
                         "[Create][StreamSwitchNode] for switch node:%s in graph:%s failed.",
                         switch_node->GetName().c_str(), graph->GetName().c_str());
        if (SetSwitchTrueBranchFlag(stream_switch, true_branch_flag) != SUCCESS) {
          REPORT_CALL_ERROR("E19999", "Set switch true branch flag from node:%s(%s) failed",
                            stream_switch->GetName().c_str(), stream_switch->GetType().c_str());
          GELOGE(FAILED, "[Set][SwitchTrueBranchFlag] for node %s failed.", stream_switch->GetName().c_str());
          return FAILED;
        }
        if (MarkBranches(peer_cond_anchor, stream_switch, true_branch_flag) != SUCCESS) {
          GELOGE(FAILED, "[Mark][Branches] for stream_switch %s failed.", stream_switch->GetName().c_str());
          return FAILED;
        }

        if (!cyclic_flag) {
          GE_CHK_STATUS(GraphUtils::AddEdge(peer_data_anchor->GetOwnerNode()->GetOutControlAnchor(),
                                            stream_switch->GetInControlAnchor()),
                        "[Add][ControlEdge] between %s and %s failed.",
                        peer_data_anchor->GetOwnerNode()->GetName().c_str(), stream_switch->GetName().c_str());
        }
      });

      GE_CHK_STATUS(GraphUtils::RemoveEdge(out_data_anchor, peer_in_anchor),
                    "[Remove][Edge] between %s and %s failed.",
                    switch_node->GetName().c_str(), peer_in_anchor->GetOwnerNode()->GetName().c_str());

      NodePtr out_node = peer_in_anchor->GetOwnerNode();
      GE_CHK_STATUS(GraphUtils::AddEdge(peer_data_anchor, peer_in_anchor),
                    "[Add][Edge] between %s and %s failed.",
                    peer_data_anchor->GetOwnerNode()->GetName().c_str(), out_node->GetName().c_str());
      GE_CHK_STATUS(GraphUtils::AddEdge(stream_switch->GetOutControlAnchor(), out_node->GetInControlAnchor()),
                    "[Add][ControlEdge] between %s and %s failed.",
                    stream_switch->GetName().c_str(), out_node->GetName().c_str());
      out_node_list.insert(out_node->GetName());
    }

    GE_IF_BOOL_EXEC(stream_switch != nullptr, {
      MoveCtrlEdges(switch_node, stream_switch);
      switch_node_map_[stream_switch] = out_node_list;
      if (SetOriginalNodeName(stream_switch, switch_node->GetName()) != SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Set original node name:%s to node:%s(%s) failed", switch_node->GetName().c_str(),
                          stream_switch->GetName().c_str(), stream_switch->GetType().c_str());
        GELOGE(FAILED, "[Set][OriginalNodeName] for node %s failed.", stream_switch->GetName().c_str());
        return FAILED;
      }
    });
  }

  (void)bypass_nodes_.insert(switch_node);
  return SUCCESS;
}

///
/// @brief Bypass Switch Node
/// @param [in] switch_node
/// @param [out] peer_data_anchor
/// @param [out] peer_cond_anchor
/// @return Status
///
Status SwitchToStreamSwitchPass::BypassSwitchNode(const NodePtr &switch_node, OutDataAnchorPtr &peer_data_anchor,
                                                  OutDataAnchorPtr &peer_cond_anchor) {
  for (uint32_t idx = 0; idx < SWITCH_INPUT_NUM; ++idx) {
    InDataAnchorPtr in_data_anchor = switch_node->GetInDataAnchor(idx);
    GE_CHECK_NOTNULL(in_data_anchor);
    OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor);
    // Remove Switch data input.
    if (GraphUtils::RemoveEdge(peer_out_anchor, in_data_anchor) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Remove edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%u) failed",
                        peer_out_anchor->GetOwnerNode()->GetName().c_str(),
                        peer_out_anchor->GetOwnerNode()->GetType().c_str(), peer_out_anchor->GetIdx(),
                        switch_node->GetName().c_str(), switch_node->GetType().c_str(), idx);
      GELOGE(FAILED, "[Remove][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%u) failed",
             peer_out_anchor->GetOwnerNode()->GetName().c_str(),
             peer_out_anchor->GetOwnerNode()->GetType().c_str(), peer_out_anchor->GetIdx(),
             switch_node->GetName().c_str(), switch_node->GetType().c_str(), idx);
      return FAILED;
    }

    if (idx == SWITCH_DATA_INPUT) {
      peer_data_anchor = peer_out_anchor;
    } else {
      peer_cond_anchor = peer_out_anchor;
    }
  }

  return SUCCESS;
}

///
/// @brief Find Switch cond input
/// @param [out] peer_cond_anchor
/// @return Status
///
Status SwitchToStreamSwitchPass::FindSwitchCondInput(OutDataAnchorPtr &peer_cond_anchor) {
  NodePtr tmp_node = nullptr;
  std::string type;
  bool pass_flag = true;
  while (pass_flag) {
    if (tmp_node == nullptr) {
      tmp_node = peer_cond_anchor->GetOwnerNode();
    } else {
      InDataAnchorPtr in_data_anchor = tmp_node->GetInDataAnchor(SWITCH_DATA_INPUT);
      GE_CHECK_NOTNULL(in_data_anchor);
      peer_cond_anchor = in_data_anchor->GetPeerOutAnchor();
      GE_CHECK_NOTNULL(peer_cond_anchor);
      tmp_node = peer_cond_anchor->GetOwnerNode();
    }

    GE_CHK_STATUS_RET(GetOriginalType(tmp_node, type), "[Get][OriginalType] failed.");
    pass_flag = ((type == SWITCH) || (type == REFSWITCH));
  }

  return SUCCESS;
}

///
/// @brief Create StreamSwitch Node
/// @param [in] graph
/// @param [in] switch_node
/// @param [in] suffix
/// @param [in] peer_cond_anchor
/// @return ge::NodePtr
///
NodePtr SwitchToStreamSwitchPass::CreateStreamSwitchNode(const ComputeGraphPtr &graph, const NodePtr &switch_node,
                                                         const std::string &suffix,
                                                         const OutDataAnchorPtr &peer_cond_anchor) {
  OpDescPtr switch_op_desc = switch_node->GetOpDesc();
  GE_CHK_BOOL_EXEC(switch_op_desc != nullptr,
                   REPORT_INNER_ERROR("E19999", "OpDesc in node is nullptr, check invalid");
                   return nullptr, "[Get][OpDesc] failed, OpDesc of Switch node is invalid.");
  GE_IF_BOOL_EXEC(switch_op_desc->GetInputsSize() != SWITCH_INPUT_NUM, {
    REPORT_INNER_ERROR("E19999", "Input desc size:%zu of node:%s(%s) not equal to %u, check invalid",
                       switch_op_desc->GetInputsSize(),
                       switch_op_desc->GetName().c_str(), switch_op_desc->GetType().c_str(), SWITCH_INPUT_NUM);
    GELOGE(FAILED, "[Check][Param] Switch input param invalid, input_size=%lu, should be %u.",
           switch_op_desc->GetInputsSize(), SWITCH_INPUT_NUM);
    return nullptr;
  });

  const std::string &node_name = switch_node->GetName() + "_" + STREAMSWITCH + suffix;
  GELOGI("Create StreamSwitch, name=%s.", node_name.c_str());
  OpDescPtr op_desc = MakeShared<OpDesc>(node_name, STREAMSWITCH);
  if (op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "New OpDesc failed");
    GELOGE(FAILED, "[New][OpDesc] failed.");
    return nullptr;
  }

  // mark hccl group id
  std::string hccl_group_id;
  if (AttrUtils::GetStr(switch_node->GetOpDesc(), ATTR_NAME_HCCL_FUSED_GROUP, hccl_group_id)) {
    (void)AttrUtils::SetStr(op_desc, ATTR_NAME_HCCL_FUSED_GROUP, hccl_group_id);
    GELOGD("Set attr ATTR_NAME_HCCL_FUSED_GROUP for Stream_Switch %s, value is %s.", node_name.c_str(),
           hccl_group_id.c_str());
  }

  int64_t switch_type;
  if (AttrUtils::GetInt(switch_node->GetOpDesc(), ATTR_NAME_STREAM_SWITCH_TYPE, switch_type)) {
    (void)AttrUtils::SetInt(op_desc, ATTR_NAME_STREAM_SWITCH_TYPE, switch_type);
    GELOGD("Set attr ATTR_NAME_STREAM_SWITCH_TYPE for Stream_Switch %s, value is %ld.", node_name.c_str(),
           switch_type);
  }

  if (!AttrUtils::SetInt(op_desc, ATTR_NAME_SWITCH_DATA_TYPE, RT_SWITCH_INT32) ||
      !AttrUtils::SetInt(op_desc, ATTR_NAME_STREAM_SWITCH_COND, (int64_t)RT_EQUAL)) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s or Attr:%s to op:%s(%s) failed",
                      ATTR_NAME_SWITCH_DATA_TYPE.c_str(), ATTR_NAME_STREAM_SWITCH_COND.c_str(),
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s or Attr:%s to op:%s(%s) failed",
           ATTR_NAME_SWITCH_DATA_TYPE.c_str(), ATTR_NAME_STREAM_SWITCH_COND.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return nullptr;
  }

  // Already checked, first input is Variable will passed, second is condition will checked.
  GeTensorDesc cond_input_desc = switch_op_desc->GetInputDesc(SWITCH_PRED_INPUT);
  GeTensorDesc input_desc(GeShape(cond_input_desc.GetShape().GetDims()), cond_input_desc.GetFormat(), DT_INT32);
  GE_CHK_BOOL_EXEC(op_desc->AddInputDesc(input_desc) == GRAPH_SUCCESS,
                   REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
                   return nullptr,
                   "[Add][InputDesc] to op:%s(%s) failed",
                   op_desc->GetName().c_str(), op_desc->GetType().c_str());
  GE_CHK_BOOL_EXEC(op_desc->AddInputDesc(input_desc) == GRAPH_SUCCESS,
                   REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
                   return nullptr,
                   "[Add][InputDesc] to op:%s(%s) failed",
                   op_desc->GetName().c_str(), op_desc->GetType().c_str());

  NodePtr stream_switch = graph->AddNode(op_desc);
  GE_CHK_BOOL_EXEC(stream_switch != nullptr,
                   REPORT_CALL_ERROR("E19999", "Add node:%s(%s) to graph:%s failed",
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
                   return nullptr,
                   "[Add][Node] %s(%s) to graph:%s failed",
                   op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
  GE_CHK_STATUS(GraphUtils::AddEdge(peer_cond_anchor, stream_switch->GetInDataAnchor(0)),
                "[Add][Edge] between %s and %s failed.",
                peer_cond_anchor->GetOwnerNode()->GetName().c_str(), stream_switch->GetName().c_str());

  int64_t group_index = -1;
  if (AttrUtils::GetInt(switch_node->GetOpDesc(), ATTR_NAME_CONTROL_FLOW_GROUP, group_index)) {
    SetControlFlowGroup(stream_switch, group_index);
  }
  return stream_switch;
}

///
/// @brief Mark Switch Branch
/// @param [in] peer_cond_anchor
/// @param [in] stream_switch
/// @param [in] true_branch_flag
/// @return Status
///
Status SwitchToStreamSwitchPass::MarkBranches(const OutDataAnchorPtr &peer_cond_anchor, const NodePtr &stream_switch,
                                              bool true_branch_flag) {
  uint32_t index = true_branch_flag ? SWITCH_TRUE_OUTPUT : SWITCH_FALSE_OUTPUT;
  auto it = cond_node_map_.find(peer_cond_anchor);
  if (it != cond_node_map_.end()) {
    int64_t switch_group_id = GetGroupId(stream_switch);
    auto switch_group_it = it->second.find(switch_group_id);
    if (switch_group_it == it->second.end()) {
      std::list<NodePtr> false_node_list;
      std::list<NodePtr> true_node_list;
      std::list<NodePtr> &node_list = true_branch_flag ? true_node_list : false_node_list;
      node_list.emplace_back(stream_switch);
      std::vector<std::list<NodePtr>> switch_list;
      switch_list.emplace_back(false_node_list);
      switch_list.emplace_back(true_node_list);
      it->second[switch_group_id] = switch_list;
    } else {
      GE_IF_BOOL_EXEC(switch_group_it->second.size() != SWITCH_OUTPUT_NUM, {
        REPORT_INNER_ERROR("E19999", "switch group size:%zu not equal to %u, group_id:%ld, check invalid",
                           switch_group_it->second.size(), SWITCH_OUTPUT_NUM, switch_group_id);
        GELOGE(INTERNAL_ERROR, "[Check][Param] switch group size:%zu not equal to %u, group_id:%ld",
               switch_group_it->second.size(), SWITCH_OUTPUT_NUM, switch_group_id);
        return FAILED;
      });
      switch_group_it->second[index].emplace_back(stream_switch);
    }
  } else {
    int64_t switch_group_id = GetGroupId(stream_switch);
    std::map<int64_t, std::vector<std::list<NodePtr>>> switch_group_map;
    std::list<NodePtr> false_node_list;
    std::list<NodePtr> true_node_list;
    std::list<NodePtr> &node_list = true_branch_flag ? true_node_list : false_node_list;
    node_list.emplace_back(stream_switch);
    std::vector<std::list<NodePtr>> switch_list;
    switch_list.emplace_back(false_node_list);
    switch_list.emplace_back(true_node_list);
    switch_group_map[switch_group_id] = switch_list;
    cond_node_map_[peer_cond_anchor] = switch_group_map;
  }
  return SUCCESS;
}

///
/// @brief Get group_id for switch_node
/// @param [in] node
/// @return group_id
///
int64_t SwitchToStreamSwitchPass::GetGroupId(const NodePtr &node) {
  std::string tailing_optimization_option;
  bool is_tailing_optimization = false;
  if (GetContext().GetOption(OPTION_EXEC_ENABLE_TAILING_OPTIMIZATION, tailing_optimization_option) == GRAPH_SUCCESS) {
    // "1" means it's True from frontend option
    is_tailing_optimization = (tailing_optimization_option == "1");
    GELOGI("Option ge.exec.isTailingOptimization is %s", tailing_optimization_option.c_str());
  }
  if (!is_tailing_optimization) {
    return 0;
  }

  std::string hccl_group_id;
  if (!AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_HCCL_FUSED_GROUP, hccl_group_id)) {
    GELOGI("Node %s can not find hccl group id.", node->GetName().c_str());
    return 0;
  }
  auto key_index = hccl_group_id.find_last_of('_');
  auto key_num = hccl_group_id.substr(key_index + 1, hccl_group_id.length() - key_index);
  GELOGI("Node:%s, hccl_group_id=%s, key_num=%s", node->GetName().c_str(), hccl_group_id.c_str(), key_num.c_str());
  int64_t num = atoi(key_num.c_str());
  if (num == 0) {
    return 0;
  }

  GELOGI("Hccl_group_id is %s, group_id is %ld", hccl_group_id.c_str(), num);
  return num;
}

///
/// @brief Combine switch nodes link to same cond
/// @param [in] graph
/// @return Status
///
Status SwitchToStreamSwitchPass::CombineSwitchNode(const ComputeGraphPtr &graph) {
  for (auto iter = cond_node_map_.begin(); iter != cond_node_map_.end(); ++iter) {
    for (auto group_iter = iter->second.begin(); group_iter != iter->second.end(); ++group_iter) {
      const std::list<NodePtr> &false_switch_list = group_iter->second[SWITCH_FALSE_OUTPUT];
      const std::list<NodePtr> &true_switch_list = group_iter->second[SWITCH_TRUE_OUTPUT];
      std::set<NodePtr> same_cond_switch;
      same_cond_switch.insert(false_switch_list.begin(), false_switch_list.end());
      same_cond_switch.insert(true_switch_list.begin(), true_switch_list.end());

      OutDataAnchorPtr peer_cond_anchor = iter->first;
      GE_CHECK_NOTNULL(peer_cond_anchor);
      NodePtr cond_node = peer_cond_anchor->GetOwnerNode();
      GELOGI("CombineSwitchNode: cond_node=%s.", cond_node->GetName().c_str());

      NodePtr cast_node = CreateCastOp(graph, peer_cond_anchor);
      GE_CHK_BOOL_EXEC(cast_node != nullptr, return FAILED,
                       "[Create][CastOp] for cond_node:%s failed.", cond_node->GetName().c_str());

      NodePtr active_node = CreateActiveNode(graph, cond_node);
      GE_CHK_BOOL_EXEC(active_node != nullptr, return FAILED,
                       "[Create][StreamActiveNode] for cond node:%s failed.", cond_node->GetName().c_str());
      GE_CHK_STATUS(GraphUtils::AddEdge(cast_node->GetOutControlAnchor(), active_node->GetInControlAnchor()),
                    "[Add][ControlEdge] between %s and %s failed.",
                    cond_node->GetName().c_str(), active_node->GetName().c_str());
      if (SetActiveLabelList(active_node, { cast_node->GetName() }) != SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Set active label list:%s to op:%s(%s) failed",
                          cast_node->GetName().c_str(), active_node->GetName().c_str(), active_node->GetType().c_str());
        GELOGE(FAILED, "[Set][ActiveLabelList] %s to op:%s(%s) failed.",
               cast_node->GetName().c_str(), active_node->GetName().c_str(), active_node->GetType().c_str());
        return FAILED;
      }

      int64_t group_index = -1;
      std::function<bool(const NodePtr &)> callback = [&group_index](const NodePtr &n) {
        return AttrUtils::GetInt(n->GetOpDesc(), ATTR_NAME_CONTROL_FLOW_GROUP, group_index);
      };
      (void)std::any_of(same_cond_switch.begin(), same_cond_switch.end(), callback);
      SetControlFlowGroup(active_node, group_index);

      const std::string &cond_group = cond_node->GetName();
      for (uint32_t i = 0; i < SWITCH_OUTPUT_NUM; ++i) {
        bool true_branch_flag = (i == SWITCH_TRUE_OUTPUT);
        const std::list<NodePtr> &switch_list = (true_branch_flag ? true_switch_list : false_switch_list);
        GE_IF_BOOL_EXEC(switch_list.empty(), continue);

        // select first stream_switch
        NodePtr stream_switch = switch_list.front();
        OpDescPtr switch_desc = stream_switch->GetOpDesc();
        GE_CHECK_NOTNULL(switch_desc);
        // set stream_label
        if (SetStreamLabel(stream_switch, cast_node->GetName()) != SUCCESS) {
          REPORT_CALL_ERROR("E19999", "Set stream_label:%s to op:%s(%s) failed",
                            cast_node->GetName().c_str(), stream_switch->GetName().c_str(),
                            stream_switch->GetType().c_str());
          GELOGE(FAILED, "[Set][StreamLabel] %s to op:%s(%s) failed", cast_node->GetName().c_str(),
                 stream_switch->GetName().c_str(), stream_switch->GetType().c_str());
          return FAILED;
        }
        switch_desc->SetName(CheckDuplicateName(cond_group + "/" + STREAMSWITCH + (true_branch_flag ? "_t" : "_f")));
        stream_switch_nodes_.emplace_back(stream_switch);

        // 0_input: original pred input, 1_input: constant node
        GE_CHK_STATUS_RET(AddConstNode(graph, stream_switch),
                          "[Add][ConstNode] failed, stream_switch:%s.", stream_switch->GetName().c_str());
        GE_CHK_STATUS(GraphUtils::RemoveEdge(peer_cond_anchor, stream_switch->GetInDataAnchor(0)),
                      "[Remove][Edge] between %s and %s failed.",
                      peer_cond_anchor->GetOwnerNode()->GetName().c_str(), stream_switch->GetName().c_str());
        GE_CHK_STATUS(GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), stream_switch->GetInDataAnchor(0)),
                      "[Add][Edge] between %s and %s failed.",
                      cast_node->GetName().c_str(), stream_switch->GetName().c_str());

        SetControlFlowGroup(stream_switch, group_index);
        for (const NodePtr &node : switch_list) {
          GE_IF_BOOL_EXEC(node != stream_switch, {
            GE_CHK_STATUS(GraphUtils::RemoveEdge(peer_cond_anchor, node->GetInDataAnchor(0)),
                          "[Remove][Edge] between %s and %s failed.",
                          peer_cond_anchor->GetOwnerNode()->GetName().c_str(), node->GetName().c_str());
          });
          GE_CHK_STATUS(ModifySwitchInCtlEdges(node, cast_node, same_cond_switch),
                        "[Modify][SwitchInCtlEdges] failed, switch node:%s, cast node:%s.",
                        node->GetName().c_str(), cast_node->GetName().c_str());
          GE_CHK_STATUS(ModifySwitchOutCtlEdges(node, stream_switch, active_node),
                        "[Modify][SwitchOutCtlEdges] failed, node:%s, stream_switch:%s.",
                        node->GetName().c_str(), stream_switch->GetName().c_str());
        }

        GE_CHK_STATUS(GraphUtils::AddEdge(active_node->GetOutControlAnchor(), stream_switch->GetInControlAnchor()),
                      "[Add][ControlEdge] between %s and %s failed.",
                      active_node->GetName().c_str(), stream_switch->GetName().c_str());
      }
    }
  }
  return SUCCESS;
}

///
/// @brief Create Active Op
/// @param [in] graph
/// @param [in] cond_node
/// @return ge::NodePtr
///
NodePtr SwitchToStreamSwitchPass::CreateActiveNode(const ComputeGraphPtr &graph, const NodePtr &node) {
  const std::string &node_name = CheckDuplicateName(node->GetName() + "_" + STREAMACTIVE);
  GELOGI("Create StreamActive op:%s.", node_name.c_str());
  OpDescPtr op_desc = MakeShared<OpDesc>(node_name, STREAMACTIVE);
  if (op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "New OpDesc failed");
    GELOGE(FAILED, "[New][OpDesc] failed.");
    return nullptr;
  }

  NodePtr active_node = graph->AddNode(op_desc);
  GE_CHK_BOOL_EXEC(active_node != nullptr,
                   REPORT_CALL_ERROR("E19999", "Add node:%s(%s) to graph:%s failed",
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
                   return nullptr,
                   "[Add][Node] %s(%s) to graph:%s failed",
                   op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());

  GE_IF_BOOL_EXEC(SetSwitchBranchNodeLabel(active_node, node_name) != SUCCESS,
                  REPORT_CALL_ERROR("E19999", "Set switch branch node label:%s to node:%s(%s) failed",
                                    node_name.c_str(), active_node->GetName().c_str(), active_node->GetType().c_str());
                  GELOGE(INTERNAL_ERROR, "[Set][SwitchBranchNodeLabel] %s to node:%s(%s) failed",
                         node_name.c_str(), active_node->GetName().c_str(), active_node->GetType().c_str());
                  return nullptr);

  return active_node;
}

///
/// @brief Create cast node
/// @param [in] graph
/// @param [in] peer_cond_anchor
/// @return NodePtr
///
NodePtr SwitchToStreamSwitchPass::CreateCastOp(const ComputeGraphPtr &graph, const OutDataAnchorPtr &peer_cond_anchor) {
  OpDescPtr cond_desc = peer_cond_anchor->GetOwnerNode()->GetOpDesc();
  GE_CHK_BOOL_EXEC(cond_desc != nullptr, return nullptr,
                   "[Get][OpDesc] failed, opdesc of Param peer_cond_anchor's owner node is nullptr.");

  const std::string &cast_name = CheckDuplicateName(cond_desc->GetName() + "_" + CAST);
  GELOGI("Create cast_node: %s, input datatype:DT_BOOL, out datatype:DT_INT32", cast_name.c_str());
  OpDescPtr cast_desc = MakeShared<OpDesc>(cast_name, CAST);
  if (cast_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "New OpDesc failed");
    GELOGE(FAILED, "[New][OpDesc] failed.");
    return nullptr;
  }
  if (!(AttrUtils::SetInt(cast_desc, CAST_ATTR_SRCT, (int64_t)DT_BOOL) &&
        AttrUtils::SetInt(cast_desc, CAST_ATTR_DSTT, (int64_t)DT_INT32) &&
        AttrUtils::SetInt(cast_desc, CAST_ATTR_DST_TYPE, (int64_t)DT_INT32) &&
        AttrUtils::SetBool(cast_desc, CAST_ATTR_TRUNCATE, false))) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s or %s or %s or %s to op:%s(%s) failed",
                      CAST_ATTR_SRCT.c_str(), CAST_ATTR_DSTT.c_str(),
                      CAST_ATTR_DST_TYPE.c_str(), CAST_ATTR_TRUNCATE.c_str(),
                      cast_desc->GetName().c_str(), cast_desc->GetType().c_str());
    GELOGE(FAILED, "[Set][Attr] %s or %s or %s or %s to op:%s(%s) failed",
           CAST_ATTR_SRCT.c_str(), CAST_ATTR_DSTT.c_str(),
           CAST_ATTR_DST_TYPE.c_str(), CAST_ATTR_TRUNCATE.c_str(),
           cast_desc->GetName().c_str(), cast_desc->GetType().c_str());
    return nullptr;
  }

  GeTensorDesc tensor_desc = cond_desc->GetOutputDesc(peer_cond_anchor->GetIdx());
  tensor_desc.SetDataType(DT_BOOL);
  GE_CHK_BOOL_EXEC(cast_desc->AddInputDesc(tensor_desc) == SUCCESS,
                   REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                                     cast_desc->GetName().c_str(), cast_desc->GetType().c_str());
                   return nullptr,
                   "[Add][InputDesc] to op:%s(%s) failed",
                   cast_desc->GetName().c_str(), cast_desc->GetType().c_str());
  tensor_desc.SetDataType(DT_INT32);
  GE_CHK_BOOL_EXEC(cast_desc->AddOutputDesc(tensor_desc) == SUCCESS,
                   REPORT_CALL_ERROR("E19999", "Add output desc to op:%s(%s) failed",
                                     cast_desc->GetName().c_str(), cast_desc->GetType().c_str());
                   return nullptr,
                   "[Add][OutputDesc] to op:%s(%s) failed",
                   cast_desc->GetName().c_str(), cast_desc->GetType().c_str());

  NodePtr cast_node = graph->AddNode(cast_desc);
  GE_CHK_BOOL_EXEC(cast_node != nullptr,
                   REPORT_CALL_ERROR("E19999", "Add node:%s(%s) to graph:%s failed",
                                     cast_desc->GetName().c_str(), cast_desc->GetType().c_str(),
                                     graph->GetName().c_str());
                   return nullptr,
                   "[Add][Node] %s(%s) to graph:%s failed",
                   cast_desc->GetName().c_str(), cast_desc->GetType().c_str(),
                   graph->GetName().c_str());
  // Cast node has and only has one input
  GE_CHK_STATUS(GraphUtils::AddEdge(peer_cond_anchor, cast_node->GetInDataAnchor(0)),
                "[Add][Edge] between %s and %s failed.",
                cond_desc->GetName().c_str(), cast_node->GetName().c_str());

  return cast_node;
}

///
/// @brief Add const node as switch input1
/// @param [in] graph
/// @param [in] stream_switch
/// @return Status
///
Status SwitchToStreamSwitchPass::AddConstNode(const ComputeGraphPtr &graph, const NodePtr &stream_switch) {
  OpDescPtr op_desc = stream_switch->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  bool value = false;
  GE_CHK_BOOL_EXEC(AttrUtils::GetBool(op_desc, ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG, value),
                   REPORT_CALL_ERROR("E19999", "Get Attr:%s from op:%s(%s) failed",
                                     ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG.c_str(),
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
                   return FAILED,
                   "[Get][Attr] %s from op:%s(%s) failed", ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG.c_str(),
                   op_desc->GetName().c_str(), op_desc->GetType().c_str());

  const std::string &const_node_name = op_desc->GetName() + "_Constant_" + (value ? "t" : "f");
  GELOGI("Create const op: %s", const_node_name.c_str());
  OpDescPtr const_op_desc = MakeShared<OpDesc>(const_node_name, CONSTANT);
  if (const_op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "New OpDesc failed");
    GELOGE(FAILED, "New OpDesc failed.");
    return FAILED;
  }

  auto resize_value = (int32_t)value;
  GeTensorDesc data_desc = op_desc->GetInputDesc(1);
  GeTensorPtr const_value =
          MakeShared<GeTensor>(data_desc, reinterpret_cast<uint8_t *>(&resize_value), sizeof(int32_t));
  if (const_value == nullptr) {
    REPORT_CALL_ERROR("E19999", "New GeTensor failed");
    GELOGE(FAILED, "[New][GeTensor] failed.");
    return FAILED;
  }
  GE_CHK_BOOL_EXEC(AttrUtils::SetTensor(const_op_desc, ATTR_NAME_WEIGHTS, const_value),
                   REPORT_CALL_ERROR("E19999", "Get Attr:%s from op:%s(%s) failed", ATTR_NAME_WEIGHTS.c_str(),
                                     const_op_desc->GetName().c_str(), const_op_desc->GetType().c_str());
                   return FAILED,
                   "[Get][Attr] %s from op:%s(%s) failed", ATTR_NAME_WEIGHTS.c_str(),
                   const_op_desc->GetName().c_str(), const_op_desc->GetType().c_str());
  GE_CHK_BOOL_EXEC(const_op_desc->AddOutputDesc(data_desc) == GRAPH_SUCCESS,
                   REPORT_CALL_ERROR("E19999", "Add output desc to op:%s(%s) failed",
                                     const_op_desc->GetName().c_str(), const_op_desc->GetType().c_str());
                   return FAILED,
                   "[Add][OutputDesc] to op:%s(%s) failed",
                   const_op_desc->GetName().c_str(), const_op_desc->GetType().c_str());

  NodePtr const_node = graph->AddNode(const_op_desc);
  GE_CHK_BOOL_EXEC(const_node != nullptr,
                   REPORT_CALL_ERROR("E19999", "Add node:%s(%s) to graph:%s failed",
                                     const_op_desc->GetName().c_str(), const_op_desc->GetType().c_str(),
                                     graph->GetName().c_str());
                   return FAILED,
                   "[Add][Node] %s(%s) to graph:%s failed",
                   const_op_desc->GetName().c_str(), const_op_desc->GetType().c_str(),
                   graph->GetName().c_str());
  GE_CHK_STATUS(GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), stream_switch->GetInDataAnchor(1)),
                "[Add][Edge] between %s and %s failed.",
                const_node->GetName().c_str(), stream_switch->GetName().c_str());

  return SUCCESS;
}

///
/// @brief Modify in ctl edge for switch_node
/// @param [in] switch_node
/// @param [in] cast_node
/// @param [in] same_cond_switch
/// @return Status
///
Status SwitchToStreamSwitchPass::ModifySwitchInCtlEdges(const NodePtr &switch_node, const NodePtr &cast_node,
                                                        const std::set<NodePtr> &same_cond_switch) {
  GELOGD("ModifySwitchInCtlEdges: switch_node=%s, cast_node=%s", switch_node->GetName().c_str(),
         cast_node->GetName().c_str());
  std::string orig_switch_name = switch_node->GetName();
  OpDescPtr switch_desc = switch_node->GetOpDesc();
  GE_CHECK_NOTNULL(switch_desc);
  if (!AttrUtils::GetStr(switch_desc, ATTR_NAME_ORIG_NODE_NAME, orig_switch_name) || orig_switch_name.empty()) {
    REPORT_CALL_ERROR("E19999", "Get Attr:%s from op:%s(%s) failed", ATTR_NAME_ORIG_NODE_NAME.c_str(),
                      switch_desc->GetName().c_str(), switch_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][Attr] %s from op:%s(%s) failed", ATTR_NAME_ORIG_NODE_NAME.c_str(),
           switch_desc->GetName().c_str(), switch_desc->GetType().c_str());
    return INTERNAL_ERROR;
  }

  for (const NodePtr &in_ctrl_node : switch_node->GetInControlNodes()) {
    GE_CHK_STATUS(GraphUtils::RemoveEdge(in_ctrl_node->GetOutControlAnchor(), switch_node->GetInControlAnchor()),
                  "[Remove][ControlEdge] between %s and %s failed.",
                  in_ctrl_node->GetName().c_str(), switch_node->GetName().c_str());
    GE_IF_BOOL_EXEC(!in_ctrl_node->GetOutControlAnchor()->IsLinkedWith(cast_node->GetInControlAnchor()), {
      GE_CHK_STATUS(GraphUtils::AddEdge(in_ctrl_node->GetOutControlAnchor(), cast_node->GetInControlAnchor()),
                    "[Add][ControlEdge] between %s and %s failed.",
                    in_ctrl_node->GetName().c_str(), cast_node->GetName().c_str());
    });

    GE_IF_BOOL_EXEC(in_ctrl_node->GetType() != STREAMSWITCH, continue);
    if (same_cond_switch.count(in_ctrl_node) > 0) {
      GE_CHK_STATUS(GraphUtils::RemoveEdge(in_ctrl_node->GetOutControlAnchor(), cast_node->GetInControlAnchor()),
                    "[Remove][ControlEdge] between %s and %s failed.",
                    in_ctrl_node->GetName().c_str(), cast_node->GetName().c_str());
      continue;
    }

    auto find_res1 = switch_node_map_.find(in_ctrl_node);
    GE_IF_BOOL_EXEC(find_res1 == switch_node_map_.end(), {
      REPORT_INNER_ERROR("E19999", "Node:%s(%s) can't find in switch_node_map_, check invalid",
                         in_ctrl_node->GetName().c_str(), in_ctrl_node->GetType().c_str());
      GELOGE(INTERNAL_ERROR, "[Check][Param] StreamSwitch node %s not found in switch_node_map_.",
             in_ctrl_node->GetName().c_str());
      return INTERNAL_ERROR;
    });
    auto find_res2 = find_res1->second.find(orig_switch_name);
    auto find_res3 = find_res1->second.find(cast_node->GetName());
    GE_IF_BOOL_EXEC((find_res2 != find_res1->second.end()) && (find_res3 == find_res1->second.end()), {
      find_res1->second.erase(find_res2);
      find_res1->second.insert(cast_node->GetName());
      continue;
    });
  }

  return SUCCESS;
}

///
/// @brief Modify out ctl edge for switch_node
/// @param [in] switch_node
/// @param [in] stream_switch
/// @param [in] active_node
/// @return Status
///
Status SwitchToStreamSwitchPass::ModifySwitchOutCtlEdges(const NodePtr &switch_node, const NodePtr &stream_switch,
                                                         const NodePtr &active_node) {
  GELOGD("ModifySwitchOutCtlEdges: switch_node=%s, stream_switch=%s, active_node=%s", switch_node->GetName().c_str(),
         stream_switch->GetName().c_str(), active_node->GetName().c_str());
  auto find_res = switch_node_map_.find(switch_node);
  GE_IF_BOOL_EXEC(find_res == switch_node_map_.end(), {
    REPORT_INNER_ERROR("E19999", "Node:%s(%s) can't find in switch_node_map_, check invalid",
                       switch_node->GetName().c_str(), switch_node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] StreamSwitch node %s not found in switch_node_map_.",
           switch_node->GetName().c_str());
    return INTERNAL_ERROR;
  });
  GE_IF_BOOL_EXEC(find_res->second.empty(), {
    REPORT_INNER_ERROR("E19999", "True_nodes of StreamSwitch node:%s(%s) is empty, check invalid",
                       switch_node->GetName().c_str(), switch_node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] true_nodes of StreamSwitch node %s is empty.",
           switch_node->GetName().c_str());
    return INTERNAL_ERROR;
  });

  for (const NodePtr &node : switch_node->GetOutControlNodes()) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    GE_CHK_STATUS(GraphUtils::RemoveEdge(switch_node->GetOutControlAnchor(), node->GetInControlAnchor()),
                  "[Remove][ControlEdge] between %s and %s failed.",
                  switch_node->GetName().c_str(), node->GetName().c_str());
    std::string orig_name = op_desc->GetName();
    GE_IF_BOOL_EXEC(op_desc->HasAttr(ATTR_NAME_ORIG_NODE_NAME), {
      if (!AttrUtils::GetStr(op_desc, ATTR_NAME_ORIG_NODE_NAME, orig_name) || orig_name.empty()) {
        REPORT_CALL_ERROR("E19999", "Get Attr:%s from op:%s(%s) failed", ATTR_NAME_ORIG_NODE_NAME.c_str(),
                          op_desc->GetName().c_str(), op_desc->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Get][Attr] %s from op:%s(%s) failed", ATTR_NAME_ORIG_NODE_NAME.c_str(),
               op_desc->GetName().c_str(), op_desc->GetType().c_str());
        return INTERNAL_ERROR;
      }
    });
    if (find_res->second.find(orig_name) == find_res->second.end()) {
      auto active_out_ctrl_anchor = active_node->GetOutControlAnchor();
      GE_CHECK_NOTNULL(active_out_ctrl_anchor);
      GE_IF_BOOL_EXEC(!active_out_ctrl_anchor->IsLinkedWith(node->GetInControlAnchor()), {
        GE_CHK_STATUS(GraphUtils::AddEdge(active_out_ctrl_anchor, node->GetInControlAnchor()),
                      "[Add][ControlEdge] between %s and %s failed.",
                      active_node->GetName().c_str(), node->GetName().c_str());
      });
    } else {
      auto switch_out_ctrl_anchor = stream_switch->GetOutControlAnchor();
      GE_CHECK_NOTNULL(switch_out_ctrl_anchor);
      GE_IF_BOOL_EXEC(!switch_out_ctrl_anchor->IsLinkedWith(node->GetInControlAnchor()), {
        GE_CHK_STATUS(GraphUtils::AddEdge(switch_out_ctrl_anchor, node->GetInControlAnchor()),
                      "[Add][ControlEdge] between %s and %s failed.",
                      stream_switch->GetName().c_str(), node->GetName().c_str());
      });
    }
  }

  GE_IF_BOOL_EXEC(switch_node != stream_switch, (void)bypass_nodes_.insert(switch_node));
  return SUCCESS;
}

///
/// @brief Check duplicate node_name
/// @param [in] node_name
/// @return std::string
///
std::string SwitchToStreamSwitchPass::CheckDuplicateName(const std::string &node_name) {
  std::string tmp_name = node_name;
  auto iter = node_num_map_.find(tmp_name);
  if (iter != node_num_map_.end()) {
    tmp_name = tmp_name + "_" + std::to_string(iter->second);
    (iter->second)++;
  } else {
    node_num_map_[tmp_name] = 1;
  }
  return tmp_name;
}

///
/// @brief Move Control Edges
/// @param [in] old_node
/// @param [in] new_node
/// @return void
///
void SwitchToStreamSwitchPass::MoveCtrlEdges(const NodePtr &old_node, const NodePtr &new_node) {
  GE_IF_BOOL_EXEC(old_node == new_node, return );
  auto iter = switch_cyclic_map_.find(old_node);
  bool check_flag = (iter != switch_cyclic_map_.end());
  for (const NodePtr &in_node : old_node->GetInControlNodes()) {
    auto out_ctrl_anchor = in_node->GetOutControlAnchor();
    GE_CHECK_NOTNULL_JUST_RETURN(out_ctrl_anchor);
    if (check_flag && (iter->second.count(in_node->GetName()) > 0)) {
      for (const auto &out_node : old_node->GetOutAllNodes()) {
        GE_IF_BOOL_EXEC(!out_ctrl_anchor->IsLinkedWith(out_node->GetInControlAnchor()), {
          GE_CHK_STATUS(GraphUtils::AddEdge(out_ctrl_anchor, out_node->GetInControlAnchor()),
                        "[Add][ControlEdge] between %s and %s failed.",
                        in_node->GetName().c_str(), out_node->GetName().c_str());
        });
      }
    } else {
      GE_IF_BOOL_EXEC(!out_ctrl_anchor->IsLinkedWith(new_node->GetInControlAnchor()), {
        GE_CHK_STATUS(GraphUtils::AddEdge(out_ctrl_anchor, new_node->GetInControlAnchor()),
                      "[Add][ControlEdge] between %s and %s failed.",
                      in_node->GetName().c_str(), new_node->GetName().c_str());
      });
    }
    GE_CHK_STATUS(GraphUtils::RemoveEdge(out_ctrl_anchor, old_node->GetInControlAnchor()),
                  "[Remove][ControlEdge] between %s and %s failed.",
                  in_node->GetName().c_str(), old_node->GetName().c_str());
  }

  for (const NodePtr &out_node : old_node->GetOutControlNodes()) {
    GE_IF_BOOL_EXEC(!new_node->GetOutControlAnchor()->IsLinkedWith(out_node->GetInControlAnchor()), {
      GE_CHK_STATUS(GraphUtils::AddEdge(new_node->GetOutControlAnchor(), out_node->GetInControlAnchor()),
                    "[Add][ControlEdge] between %s and %s failed.",
                    new_node->GetName().c_str(), out_node->GetName().c_str());
    });
    GE_CHK_STATUS(GraphUtils::RemoveEdge(old_node->GetOutControlAnchor(), out_node->GetInControlAnchor()),
                  "[Remove][ControlEdge] between %s and %s failed.",
                  old_node->GetName().c_str(), out_node->GetName().c_str());
  }
}
}  // namespace ge
