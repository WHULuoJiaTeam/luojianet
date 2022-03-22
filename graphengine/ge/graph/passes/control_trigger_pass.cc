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

#include "graph/passes/control_trigger_pass.h"
#include <stack>
#include "common/ge/ge_util.h"
#include "common/omg_util.h"
#include "graph/utils/type_utils.h"

namespace ge {
Status ControlTriggerPass::Run(ComputeGraphPtr graph) {
  GELOGD("ControlTriggerPass Enter");
  for (NodePtr &node : graph->GetDirectNode()) {
    if (node->GetType() != CONTROLTRIGGER) {
      continue;
    }
    auto in_ctrl_nodes = node->GetInControlNodes();
    for (NodePtr &in_ctrl_node : in_ctrl_nodes) {
      if (HandleDynamicCtrlEdges(graph, node, in_ctrl_node) != SUCCESS) {
        GELOGE(FAILED, "[Handle][DynamicCtrlEdges] for node:%s->node:%s failed.", in_ctrl_node->GetName().c_str(),
               node->GetName().c_str());
        return FAILED;
      }
    }
  }

  GELOGD("ControlTriggerPass Leave");
  return SUCCESS;
}

///
/// @brief Handle input ctrl edges for ControlTrigger node
/// @param [in] graph
/// @param [in] node
/// @param [in] in_ctrl_node
/// @return Status
///
Status ControlTriggerPass::HandleDynamicCtrlEdges(ComputeGraphPtr &graph, NodePtr &node, NodePtr &in_ctrl_node) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(in_ctrl_node);
  GELOGI("HandleDynamicCtrlEdges: node=%s, in_ctrl_node=%s", node->GetName().c_str(), in_ctrl_node->GetName().c_str());
  NodePtr switch_node = nullptr;
  bool branch_flag = false;
  if (FindSwitchNode(in_ctrl_node, switch_node, branch_flag) != SUCCESS) {
    GELOGE(FAILED, "[Find][SwitchNode] failed, in_ctrl_node:%s.", in_ctrl_node->GetName().c_str());
    return FAILED;
  }

  if (switch_node == nullptr) {
    GELOGI("Not find valid switch node.");
    return SUCCESS;
  }
  auto iter1 = control_trigger_map_.find(node);
  if (iter1 != control_trigger_map_.end()) {
    auto iter2 = iter1->second.find(switch_cond_map_[switch_node]);
    if (iter2 != iter1->second.end()) {
      NodePtr constant = (branch_flag ? iter2->second.second : iter2->second.first);
      if ((GraphUtils::RemoveEdge(in_ctrl_node->GetOutControlAnchor(), node->GetInControlAnchor()) != GRAPH_SUCCESS) ||
          (GraphUtils::AddEdge(in_ctrl_node->GetOutControlAnchor(), constant->GetInControlAnchor()) != GRAPH_SUCCESS)) {
        REPORT_CALL_ERROR("E19999", "Remove control edge between op:%s(%s) and op:%s(%s), then "
                          "add control edge between op:%s(%s) and op:%s(%s) failed",
                          in_ctrl_node->GetName().c_str(), in_ctrl_node->GetType().c_str(),
                          node->GetName().c_str(), node->GetType().c_str(),
                          in_ctrl_node->GetName().c_str(), in_ctrl_node->GetType().c_str(),
                          constant->GetName().c_str(), constant->GetType().c_str());
        GELOGE(FAILED, "[Replace][CtrlEdge] failed, remove edge:%s->%s, add edge:%s->%s.",
               in_ctrl_node->GetName().c_str(), node->GetName().c_str(),
               in_ctrl_node->GetName().c_str(), constant->GetName().c_str());
        return FAILED;
      }

      GELOGI("No need to insert new branch.");
      return SUCCESS;
    }
  }

  if (InsertOppositeBranch(graph, node, in_ctrl_node, switch_node, branch_flag) != SUCCESS) {
    GELOGE(FAILED, "[Insert][OppositeBranch] failed, node:%s, in_ctrl_node:%s.",
           node->GetName().c_str(), in_ctrl_node->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief Find switch_node for ControlTrigger node
/// @param [in] node
/// @param [out] switch_node
/// @param [out] branch_flag
/// @return Status
///
Status ControlTriggerPass::FindSwitchNode(const NodePtr &node, NodePtr &switch_node, bool &branch_flag) {
  std::set<std::pair<NodePtr, uint32_t>> handle_nodes;
  // {node, <idx, <cond_merge_num, loop_switchf_num>>}
  std::stack<std::pair<NodePtr, std::pair<uint32_t, std::pair<uint32_t, uint32_t>>>> nodes;
  nodes.push(std::make_pair(node, std::make_pair(UINT32_MAX, std::make_pair(0, 0))));
  std::set<std::pair<NodePtr, uint32_t>> in_nodes;

  while (!nodes.empty()) {
    auto iter = nodes.top();
    NodePtr tmp_node = iter.first;
    GE_CHECK_NOTNULL(tmp_node);
    nodes.pop();
    uint32_t index = iter.second.first;
    auto num_pair = iter.second.second;
    if (handle_nodes.count(std::make_pair(tmp_node, index)) > 0) {
      continue;
    }
    switch (TransferNodeType(tmp_node, index)) {
      case kCondSwitch:
        if (num_pair.first == 0) {
          switch_node = tmp_node;
          branch_flag = (index == SWITCH_TRUE_OUTPUT);
          GELOGI("FindSwitchNode succ, switch_node=%s, idx=%u", switch_node->GetName().c_str(), index);
          return SUCCESS;
        }
        num_pair.first--;
        break;
      case kCondMerge:
        num_pair.first++;
        break;
      case kLoopSwitchT:
        GELOGI("in while_body, no need handle");
        return SUCCESS;
      case kLoopSwitchF:
        num_pair.second++;
        break;
      case kEnter:
        if (num_pair.second > 0) {
          num_pair.second--;
        }
        break;
      case kNotControlOp:
        break;
      default:
        GELOGE(FAILED, "[Check][Param] invalid node type");
        return FAILED;
    }

    GetInNodes(tmp_node, in_nodes);
    for (auto &node_idx : in_nodes) {
      nodes.push(std::make_pair(node_idx.first, std::make_pair(node_idx.second, num_pair)));
    }

    (void)handle_nodes.insert(std::make_pair(tmp_node, index));
  }

  return SUCCESS;
}

///
/// @brief Check if need insert opposite branch
/// @param [in] node
/// @param [in] index
/// @return ControlNodeType
///
ControlNodeType ControlTriggerPass::TransferNodeType(const NodePtr &node, uint32_t index) {
  OpDescPtr merge_desc = node->GetOpDesc();
  if (merge_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "op_desc in merge node is nullptr, check invalid");
    GELOGE(INTERNAL_ERROR, "[Get][OpDesc] failed, merge_desc is nullptr.");
    return kInvalidType;
  }
  const std::string type = node->GetType();
  if ((type == SWITCH) || (type == REFSWITCH)) {
    if ((index != SWITCH_TRUE_OUTPUT) && (index != SWITCH_FALSE_OUTPUT)) {
      GELOGI("TransferNodeType: neither true nor false branch.");
      return kNotControlOp;
    }

    if (FindPredInput(node) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "[Find][PredInput] failed, switch_node:%s.", node->GetName().c_str());
      return kInvalidType;
    }

    NodePtr pred_node = switch_cond_map_[node];
    bool branch_flag = (index == SWITCH_TRUE_OUTPUT);
    if (pred_node->GetType() != LOOPCOND) {
      GELOGI("TransferNodeType: kCondSwitch node=%s, idx=%u", node->GetName().c_str(), index);
      return kCondSwitch;
    } else {
      GELOGI("TransferNodeType: kLoopSwitch node=%s, idx=%u", node->GetName().c_str(), index);
      return branch_flag ? kLoopSwitchT : kLoopSwitchF;
    }
  } else if ((type == MERGE) || (type == REFMERGE)) {
    if (!merge_desc->HasAttr(ATTR_NAME_NEXT_ITERATION)) {
      return kCondMerge;
    }
  } else if ((type == ENTER) || (type == REFENTER)) {
    return kEnter;
  }

  return kNotControlOp;
}

///
/// @brief Get in_node & idx pairs
/// @param [in] node
/// @param [out] in_nodes
/// @return void
///
void ControlTriggerPass::GetInNodes(const NodePtr &node, std::set<std::pair<NodePtr, uint32_t>> &in_nodes) {
  in_nodes.clear();
  for (auto &in_ctrl_node : node->GetInControlNodes()) {
    (void)in_nodes.insert(std::make_pair(in_ctrl_node, UINT32_MAX));
  }

  for (InDataAnchorPtr &in_data_anchor : node->GetAllInDataAnchors()) {
    OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;
    }
    (void)in_nodes.insert(std::make_pair(peer_out_anchor->GetOwnerNode(), peer_out_anchor->GetIdx()));
  }
  return;
}

///
/// @brief Insert opposite branch for ControlTrigger
/// @param [in] graph
/// @param [in] ControlTrigger node
/// @param [in] in_ctrl_node
/// @param [in] switch_node
/// @param [in] branch_flag
/// @return Status
///
Status ControlTriggerPass::InsertOppositeBranch(ComputeGraphPtr &graph, NodePtr &node, NodePtr &in_ctrl_node,
                                                NodePtr &switch_node, bool branch_flag) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(in_ctrl_node);
  GE_CHECK_NOTNULL(switch_node);
  OpDescPtr switch_desc = switch_node->GetOpDesc();
  GE_CHECK_NOTNULL(switch_desc);

  GeTensorDesc data_desc(GeShape(), FORMAT_NCHW, DT_INT32);

  NodePtr merge_node = InsertMergeNode(graph, node, in_ctrl_node, data_desc);
  if (merge_node == nullptr) {
    GELOGE(FAILED, "[Insert][MergeNode] failed, node:%s, in_ctrl_node:%s.",
           node->GetName().c_str(), in_ctrl_node->GetName().c_str());
    return FAILED;
  }

  NodePtr const_f = InsertConstNode(graph, merge_node, data_desc, false);
  NodePtr const_t = InsertConstNode(graph, merge_node, data_desc, true);
  if ((const_f == nullptr) || (const_t == nullptr)) {
    GELOGE(FAILED, "[Insert][ConstNode] failed, graph:%s, merge_node:%s.",
           graph->GetName().c_str(), merge_node->GetName().c_str());
    return FAILED;
  }

  NodePtr orig_const = branch_flag ? const_t : const_f;
  NodePtr new_const = !branch_flag ? const_t : const_f;
  uint32_t new_idx = branch_flag ? SWITCH_FALSE_OUTPUT : SWITCH_TRUE_OUTPUT;

  const std::string identity_name = switch_desc->GetName() + "_" + IDENTITY;
  NodePtr identity_node = InsertIdentityNode(graph, identity_name, switch_desc->GetOutputDesc(new_idx));
  if (identity_node == nullptr) {
    GELOGE(FAILED, "[Insert][IdentityNode] name:%s failed, graph:%s.",
           identity_name.c_str(), graph->GetName().c_str());
    return FAILED;
  }

  if (GraphUtils::AddEdge(in_ctrl_node->GetOutControlAnchor(), orig_const->GetInControlAnchor()) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                      in_ctrl_node->GetName().c_str(), in_ctrl_node->GetType().c_str(),
                      orig_const->GetName().c_str(), orig_const->GetType().c_str());
    GELOGE(FAILED, "[Add][CtrlEdge] failed, %s->%s.", in_ctrl_node->GetName().c_str(), orig_const->GetName().c_str());
    return FAILED;
  }
  if (GraphUtils::AddEdge(switch_node->GetOutDataAnchor(new_idx), identity_node->GetInDataAnchor(0)) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%u) and op:%s(%s)(index:0) failed",
                      switch_node->GetName().c_str(), switch_node->GetType().c_str(), new_idx,
                      identity_node->GetName().c_str(), identity_node->GetType().c_str());
    GELOGE(FAILED, "[Add][Edge] between op:%s(%s)(index:%u) and op:%s(%s)(index:0) failed",
           switch_node->GetName().c_str(), switch_node->GetType().c_str(), new_idx,
           identity_node->GetName().c_str(), identity_node->GetType().c_str());
    return FAILED;
  }
  if (GraphUtils::AddEdge(identity_node->GetOutControlAnchor(), new_const->GetInControlAnchor()) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                      identity_node->GetName().c_str(), identity_node->GetType().c_str(),
                      new_const->GetName().c_str(), new_const->GetType().c_str());
    GELOGE(FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
           identity_node->GetName().c_str(), identity_node->GetType().c_str(),
           new_const->GetName().c_str(), new_const->GetType().c_str());
    return FAILED;
  }

  auto pred_const = std::make_pair(switch_cond_map_[switch_node], std::make_pair(const_f, const_t));
  auto iter = control_trigger_map_.find(node);
  if (iter == control_trigger_map_.end()) {
    control_trigger_map_[node] = {pred_const};
  } else {
    if (!iter->second.insert(pred_const).second) {
      REPORT_INNER_ERROR("E19999", "Insert to control_trigger_map_ failed");
      GELOGE(FAILED, "[Check][Param] control_trigger_map_ insert failed.");
      return FAILED;
    }
  }

  return SUCCESS;
}

///
/// @brief Insert Merge Node
/// @param [in] graph
/// @param [in] node
/// @param [in] in_ctrl_node
/// @param [in] data_desc
/// @return NodePtr
///
NodePtr ControlTriggerPass::InsertMergeNode(ComputeGraphPtr &graph, NodePtr &node, NodePtr &in_ctrl_node,
                                            const GeTensorDesc &data_desc) {
  const std::string name = node->GetName() + "_" + MERGE;
  OpDescPtr op_desc = MakeShared<OpDesc>(name, MERGE);
  if (op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "New OpDesc failed");
    GELOGE(FAILED, "[New][OpDesc] failed");
    return nullptr;
  }

  if ((op_desc->AddInputDesc(data_desc) != GRAPH_SUCCESS) || (op_desc->AddInputDesc(data_desc) != GRAPH_SUCCESS) ||
      (op_desc->AddOutputDesc(data_desc) != GRAPH_SUCCESS) || (op_desc->AddOutputDesc(data_desc) != GRAPH_SUCCESS)) {
    REPORT_CALL_ERROR("E19999", "Add input or ouput desc to op:%s(%s) failed",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][GeTensorDesc] to op:%s(%s) failed",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return nullptr;
  }

  GELOGI("Create Merge op:%s.", name.c_str());
  NodePtr merge_node = graph->AddNode(op_desc);
  if (merge_node == nullptr) {
    REPORT_CALL_ERROR("E19999", "Add node:%s(%s) to graph:%s failed",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][Node] %s(%s) to graph:%s failed",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    return nullptr;
  }

  if ((GraphUtils::RemoveEdge(in_ctrl_node->GetOutControlAnchor(), node->GetInControlAnchor()) != GRAPH_SUCCESS) ||
      (GraphUtils::AddEdge(merge_node->GetOutControlAnchor(), node->GetInControlAnchor()) != GRAPH_SUCCESS)) {
    REPORT_CALL_ERROR("E19999", "Remove control edge between op:%s(%s) and op:%s(%s), then "
                      "add control edge between op:%s(%s) and op:%s(%s) failed",
                      in_ctrl_node->GetName().c_str(), in_ctrl_node->GetType().c_str(),
                      node->GetName().c_str(), node->GetType().c_str(),
                      merge_node->GetName().c_str(), merge_node->GetType().c_str(),
                      node->GetName().c_str(), node->GetType().c_str());
    GELOGE(FAILED, "[Replace][CtrlEdge] failed, remove edge:%s->%s, add edge:%s->%s",
           in_ctrl_node->GetName().c_str(), node->GetName().c_str(),
           merge_node->GetName().c_str(), node->GetName().c_str());
    return nullptr;
  }

  return merge_node;
}

///
/// @brief Insert Const Node
/// @param [in] graph
/// @param [in] merge_node
/// @param [in] data_desc
/// @param [in] flag
/// @return NodePtr
///
NodePtr ControlTriggerPass::InsertConstNode(ComputeGraphPtr &graph, NodePtr &merge_node, const GeTensorDesc &data_desc,
                                            bool flag) {
  const std::string name = merge_node->GetName() + "_" + CONSTANT + (flag ? "_t" : "_f");
  OpDescPtr op_desc = MakeShared<OpDesc>(name, CONSTANT);
  if (op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "New OpDesc failed");
    GELOGE(FAILED, "[New][OpDesc] failed.");
    return nullptr;
  }

  int32_t value = 0;
  GeTensorPtr const_value = MakeShared<GeTensor>(data_desc, reinterpret_cast<uint8_t *>(&value), sizeof(int32_t));
  if (const_value == nullptr) {
    REPORT_CALL_ERROR("E19999", "New GeTensor failed");
    GELOGE(FAILED, "[New][GeTensor] failed.");
    return nullptr;
  }
  if (!AttrUtils::SetTensor(op_desc, ATTR_NAME_WEIGHTS, const_value)) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_WEIGHTS.c_str(),
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_WEIGHTS.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return nullptr;
  }

  if (op_desc->AddOutputDesc(data_desc) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add ouput desc to op:%s(%s) failed",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][OutputDesc] to op:%s(%s) failed",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return nullptr;
  }

  GELOGI("Create Const op: %s", name.c_str());
  NodePtr const_node = graph->AddNode(op_desc);
  if (const_node == nullptr) {
    REPORT_CALL_ERROR("E19999", "Add node:%s(%s) to graph:%s failed",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][Node] %s(%s) to graph:%s failed",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    return nullptr;
  }

  uint32_t out_idx = (flag ? SWITCH_TRUE_OUTPUT : SWITCH_FALSE_OUTPUT);
  if (GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), merge_node->GetInDataAnchor(out_idx)) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:%u) failed",
                      const_node->GetName().c_str(), const_node->GetType().c_str(),
                      merge_node->GetName().c_str(), merge_node->GetType().c_str(), out_idx);
    GELOGE(FAILED, "[Add][Edge] between op:%s(%s)(index:0) and op:%s(%s)(index:%u) failed",
           const_node->GetName().c_str(), const_node->GetType().c_str(),
           merge_node->GetName().c_str(), merge_node->GetType().c_str(), out_idx);
    return nullptr;
  }

  return const_node;
}

///
/// @brief Insert Identity Node
/// @param [in] graph
/// @param [in] name
/// @param [in] data_desc
/// @return NodePtr
///
NodePtr ControlTriggerPass::InsertIdentityNode(ComputeGraphPtr &graph, const std::string &name,
                                               const GeTensorDesc &data_desc) {
  OpDescPtr op_desc = MakeShared<OpDesc>(name, IDENTITY);
  if (op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "New OpDesc failed");
    GELOGE(FAILED, "[New][OpDesc] failed");
    return nullptr;
  }

  if ((op_desc->AddInputDesc(data_desc) != GRAPH_SUCCESS) || (op_desc->AddOutputDesc(data_desc) != GRAPH_SUCCESS)) {
    REPORT_CALL_ERROR("E19999", "Add input or output desc to op:%s(%s) failed",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][GeTensorDesc] to op:%s(%s) failed",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return nullptr;
  }

  GELOGI("Create Identity op:%s.", name.c_str());
  NodePtr identity_node = graph->AddNode(op_desc);
  if (identity_node == nullptr) {
    REPORT_CALL_ERROR("E19999", "Add node:%s(%s) to graph:%s failed",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][Node] %s(%s) to graph:%s failed",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    return nullptr;
  }

  return identity_node;
}

///
/// @brief Find pred_input of switch_node
/// @param [in] switch_node
/// @param [in] name
/// @param [in] data_desc
/// @return Status
///
Status ControlTriggerPass::FindPredInput(const NodePtr &switch_node) {
  if (switch_node == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param switch_node is nullptr, check invalid");
    GELOGE(INTERNAL_ERROR, "[Check][Param] switch_node is nullptr");
    return INTERNAL_ERROR;
  }

  InDataAnchorPtr in_cond_anchor = switch_node->GetInDataAnchor(SWITCH_PRED_INPUT);
  if (in_cond_anchor == nullptr) {
    REPORT_INNER_ERROR("E19999", "Index:%d in anchor of switch_node:%s(%s) is nullptr, check invalid",
                       SWITCH_PRED_INPUT,
                       switch_node->GetName().c_str(), switch_node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][InDataAnchor] Index:%d in anchor of switch_node:%s(%s) is nullptr",
           SWITCH_PRED_INPUT, switch_node->GetName().c_str(), switch_node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  OutDataAnchorPtr pred_cond_anchor = in_cond_anchor->GetPeerOutAnchor();
  if (pred_cond_anchor == nullptr) {
    REPORT_INNER_ERROR("E19999", "Index:%d in anchor of switch_node:%s(%s), it's peer anchor is nullptr, "
                       "check invalid", SWITCH_PRED_INPUT,
                       switch_node->GetName().c_str(), switch_node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "Index:%d in anchor of switch_node:%s(%s), it's peer anchor is nullptr",
           SWITCH_PRED_INPUT, switch_node->GetName().c_str(), switch_node->GetType().c_str());
    return INTERNAL_ERROR;
  }

  switch_cond_map_[switch_node] = pred_cond_anchor->GetOwnerNode();
  return SUCCESS;
}
///
/// @brief Clear Status, used for subgraph pass
/// @return SUCCESS
///
Status ControlTriggerPass::ClearStatus() {
  switch_cond_map_.clear();
  control_trigger_map_.clear();
  return SUCCESS;
}
}  // namespace ge
