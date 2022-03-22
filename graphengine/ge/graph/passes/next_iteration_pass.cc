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

#include "graph/passes/next_iteration_pass.h"

#include "common/ge/ge_util.h"
#include "common/omg_util.h"
#include "graph/utils/node_utils.h"

using std::string;

namespace ge {
namespace {
constexpr int64_t kLoopType = 1;
constexpr uint8_t kMaxTransOp = 3;
constexpr uint8_t kTransOpIoSize = 1;
}

Status NextIterationPass::Run(ComputeGraphPtr graph) {
  GELOGD("NextIterationPass Enter");
  /// Enter-----------+
  ///                 +-> Merge -> Switch <- LoopCond <- Cond
  /// NextIteration---+
  for (auto &node : graph->GetDirectNode()) {
    const std::string type = node->GetType();
    if ((type != ENTER) && (type != REFENTER)) {
      continue;
    }
    if (GroupEnterNode(node) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "[Group][EnterNode] %s failed.", node->GetName().c_str());
      return INTERNAL_ERROR;
    }
  }

  if (FindWhileGroups() != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Find][WhileGroups] in graph:%s failed.", graph->GetName().c_str());
    return INTERNAL_ERROR;
  }

  if (!VerifyWhileGroup()) {
    GELOGE(INTERNAL_ERROR, "[Verify][WhileGroup] in graph:%s failed.", graph->GetName().c_str());
    return INTERNAL_ERROR;
  }

  if (HandleWhileGroup(graph) != SUCCESS) {
    GELOGE(FAILED, "[Handle][WhileGroup] in graph:%s failed.", graph->GetName().c_str());
    return FAILED;
  }

  GELOGD("NextIterationPass Leave");
  return SUCCESS;
}

///
/// @brief Group Enter node
/// @param [in] enter_node
/// @return Status
///
Status NextIterationPass::GroupEnterNode(const NodePtr &enter_node) {
  OpDescPtr enter_desc = enter_node->GetOpDesc();
  GE_CHECK_NOTNULL(enter_desc);
  std::string frame_name;
  if (!ge::AttrUtils::GetStr(enter_desc, ENTER_ATTR_FRAME_NAME, frame_name) || frame_name.empty()) {
    REPORT_CALL_ERROR("E19999", "Get Attr:%s from op:%s(%s) failed", ENTER_ATTR_FRAME_NAME.c_str(),
                      enter_desc->GetName().c_str(), enter_desc->GetType().c_str());
    GELOGE(FAILED, "[Get][Attr] %s from op:%s(%s) failed", ENTER_ATTR_FRAME_NAME.c_str(),
           enter_desc->GetName().c_str(), enter_desc->GetType().c_str());
    return FAILED;
  }

  string batch_label;
  if (ge::AttrUtils::GetStr(enter_desc, ATTR_NAME_BATCH_LABEL, batch_label)) {
    frame_name += batch_label;
  }

  auto iter = loop_group_map_.find(frame_name);
  if (iter == loop_group_map_.end()) {
    LoopCondGroupPtr loop_group = MakeShared<LoopCondGroup>();
    if (loop_group == nullptr) {
      REPORT_CALL_ERROR("E19999", "New LoopCondGroup failed");
      GELOGE(FAILED, "[New][LoopCondGroup] failed.");
      return FAILED;
    }
    loop_group->enter_nodes.emplace_back(enter_node);
    loop_group_map_[frame_name] = loop_group;
  } else {
    iter->second->enter_nodes.emplace_back(enter_node);
  }

  return SUCCESS;
}

///
/// @brief Find while groups
/// @return Status
///
Status NextIterationPass::FindWhileGroups() {
  for (const auto &loop_group_iter : loop_group_map_) {
    const std::string &frame_name = loop_group_iter.first;
    for (const auto &enter_node : loop_group_iter.second->enter_nodes) {
      for (const auto &out_node : enter_node->GetOutAllNodes()) {
        std::string type;
        GE_CHK_STATUS_RET(GetOriginalType(out_node, type), "[Get][OriginalType] failed.");
        if ((type != MERGE) && (type != REFMERGE)) {
          continue;
        }

        NodePtr next_node = nullptr;
        if (FindTargetNode(out_node, NEXTITERATION, true, next_node) != SUCCESS) {
          GELOGE(INTERNAL_ERROR, "[Get][NextIterationNode] failed, frame_name:%s", frame_name.c_str());
          return INTERNAL_ERROR;
        }
        loop_group_iter.second->merge_next_pairs.emplace_back(std::make_pair(out_node, next_node));

        NodePtr switch_node = nullptr;
        if (FindTargetNode(out_node, SWITCH, false, switch_node) != SUCCESS) {
          GELOGE(INTERNAL_ERROR, "[Get][SwitchNode] failed, frame_name:%s.", frame_name.c_str());
          return INTERNAL_ERROR;
        }
        if (switch_node == nullptr) {
          continue;
        }
        if (!AttrUtils::SetInt(switch_node->GetOpDesc(), ATTR_NAME_STREAM_SWITCH_TYPE, kLoopType)) {
          REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_STREAM_SWITCH_TYPE.c_str(),
                            switch_node->GetName().c_str(), switch_node->GetType().c_str());
          GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_STREAM_SWITCH_TYPE.c_str(),
                 switch_node->GetName().c_str(), switch_node->GetType().c_str());
          return INTERNAL_ERROR;
        }
        NodePtr loop_cond = nullptr;
        if (FindTargetNode(switch_node, LOOPCOND, true, loop_cond) != SUCCESS) {
          GELOGE(INTERNAL_ERROR, "[Get][LoopCondNode] failed, frame_name:%s.", frame_name.c_str());
          return INTERNAL_ERROR;
        }
        loop_group_iter.second->switch_nodes.emplace_back(switch_node);
        if (loop_group_iter.second->loop_cond == nullptr) {
          loop_group_iter.second->loop_cond = loop_cond;
        } else if (loop_group_iter.second->loop_cond != loop_cond) {
          REPORT_INNER_ERROR("E19999", "Multi LoopCond nodes exist, frame_name:%s, check invalid", frame_name.c_str());
          GELOGE(FAILED, "[Check][Param] Multi LoopCond nodes exist, frame_name:%s.", frame_name.c_str());
          return FAILED;
        }
      }
    }
  }

  return SUCCESS;
}

///
/// @brief Verify if valid
/// @return bool
///
bool NextIterationPass::VerifyWhileGroup() {
  // map<frame_name, LoopCondGroup>
  for (const auto &loop_group_iter : loop_group_map_) {
    const std::string &frame_name = loop_group_iter.first;
    if (frame_name.empty()) {
      REPORT_INNER_ERROR("E19999", "Verify while group failed, frame_name is empty");
      GELOGE(INTERNAL_ERROR, "[Check][Param] Verify while group failed, frame_name is empty.");
      return false;
    }
    if (loop_group_iter.second->loop_cond == nullptr) {
      REPORT_INNER_ERROR("E19999", "Verify while group failed, LoopCond is null, frame_name:%s.", frame_name.c_str());
      GELOGE(INTERNAL_ERROR, "[Check][Param] Verify while group failed, LoopCond is null, frame_name:%s.",
             frame_name.c_str());
      return false;
    }

    for (const auto &pair_iter : loop_group_iter.second->merge_next_pairs) {
      if ((pair_iter.first == nullptr) || (pair_iter.second == nullptr)) {
        REPORT_INNER_ERROR("E19999", "Verify while group failed, merge_node/next_node is null, frame_name:%s.",
                           frame_name.c_str());
        GELOGE(INTERNAL_ERROR, "[Check][Param] Verify while group failed, merge_node/next_node is null, frame_name:%s.",
               frame_name.c_str());
        return false;
      }
    }
  }

  return true;
}

///
/// @brief Handle while group
/// @param [in] graph
/// @return Status
///
Status NextIterationPass::HandleWhileGroup(ComputeGraphPtr &graph) {
  for (const auto &loop_cond_iter : loop_group_map_) {
    const LoopCondGroup &loop_group = *loop_cond_iter.second;
    const std::string &cond_name = loop_cond_iter.second->loop_cond->GetName();
    const int64_t group_index = loop_group.loop_cond->GetOpDesc()->GetId();
    GELOGI("Handle while group, LoopCond node: %s.", cond_name.c_str());

    // Create Active node, Enter->Active->Merge, NextIteration->Active->Merge
    NodePtr enter_active = CreateActiveNode(graph, cond_name + "_Enter_" + STREAMACTIVE);
    NodePtr next_active = CreateActiveNode(graph, cond_name + "_Next_" + STREAMACTIVE);
    if ((enter_active == nullptr) || (next_active == nullptr)) {
      GELOGE(INTERNAL_ERROR, "[Create][ActiveNode] failed, cond_name:%s.", cond_name.c_str());
      return INTERNAL_ERROR;
    }

    for (const auto &enter_node : loop_cond_iter.second->enter_nodes) {
      // Enter --> Active
      if (GraphUtils::AddEdge(enter_node->GetOutControlAnchor(), enter_active->GetInControlAnchor()) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                          enter_node->GetName().c_str(), enter_node->GetType().c_str(),
                          enter_active->GetName().c_str(), enter_active->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
               enter_node->GetName().c_str(), enter_node->GetType().c_str(),
               enter_active->GetName().c_str(), enter_active->GetType().c_str());
        return INTERNAL_ERROR;
      }
      SetControlFlowGroup(enter_node, group_index);
    }

    for (const auto &pair : loop_cond_iter.second->merge_next_pairs) {
      NodePtr merge_node = pair.first;
      NodePtr next_node = pair.second;
      // Active --> Merge
      if (GraphUtils::AddEdge(enter_active->GetOutControlAnchor(), merge_node->GetInControlAnchor()) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                          enter_active->GetName().c_str(), enter_active->GetType().c_str(),
                          merge_node->GetName().c_str(), merge_node->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
               enter_active->GetName().c_str(), enter_active->GetType().c_str(),
               merge_node->GetName().c_str(), merge_node->GetType().c_str());
        return INTERNAL_ERROR;
      }

      // NextIteration --> Active
      if (GraphUtils::AddEdge(next_node->GetOutControlAnchor(), next_active->GetInControlAnchor()) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                          next_node->GetName().c_str(), next_node->GetType().c_str(),
                          next_active->GetName().c_str(), next_active->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
               next_node->GetName().c_str(), next_node->GetType().c_str(),
               next_active->GetName().c_str(), next_active->GetType().c_str());
        return INTERNAL_ERROR;
      }

      // break link between NextIteration and Merge
      if (BreakNextIteration(next_node, merge_node) != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "[Break][NextIteration] failed, next_node:%s, merge_node:%s",
               next_node->GetName().c_str(), merge_node->GetName().c_str());
        return INTERNAL_ERROR;
      }

      SetControlFlowGroup(next_node, group_index);
      SetControlFlowGroup(merge_node, group_index);
    }

    if ((SetActiveLabelList(enter_active, {cond_name}) != SUCCESS) ||
        (SetActiveLabelList(next_active, {cond_name}) != SUCCESS)) {
      GELOGE(INTERNAL_ERROR, "[Set][ActiveLabelList] failed, cond_name:%s.", cond_name.c_str());
      return INTERNAL_ERROR;
    }

    SetControlFlowGroup(loop_group.loop_cond, group_index);
    SetControlFlowGroup(enter_active, group_index);
    SetControlFlowGroup(next_active, group_index);
    HandleSwitchExitNodes(loop_group, group_index);
  }

  return SUCCESS;
}

///
/// @brief Mark force unknown for Exit node
/// @param [in] group of LoopCond
/// @param [in] index of LoopCond Node
/// @return void
///
void NextIterationPass::HandleSwitchExitNodes(const LoopCondGroup &loop_group, int64_t group_index) {
  std::string node_type;
  for (const auto &switch_node : loop_group.switch_nodes) {
    SetControlFlowGroup(switch_node, group_index);
    for (auto node : switch_node->GetOutDataNodes()) {
      // Switch --> Exit
      // Switch --> Cast --> Exit
      // Switch --> TransData --> Cast --> Exit
      for (uint8_t i  = 0; i < kMaxTransOp; ++i) {
        if (node->GetInDataNodes().size() != kTransOpIoSize || node->GetAllOutDataAnchorsSize() != kTransOpIoSize) {
          break;
        }

        if (kExitOpTypes.count(NodeUtils::GetNodeType(node)) > 0) {
          SetControlFlowGroup(node, group_index);
          break;
        }

        const auto &all_nodes = node->GetOutAllNodes();
        if (all_nodes.size() != kTransOpIoSize) {
          break;
        }
        node = all_nodes.at(0);
      }
    }
  }
}

///
/// @brief Create Active Node
/// @param [in] graph
/// @param [in] name
/// @return ge::NodePtr
///
NodePtr NextIterationPass::CreateActiveNode(ComputeGraphPtr &graph, const std::string &name) {
  OpDescPtr op_desc = MakeShared<OpDesc>(name, STREAMACTIVE);
  if (op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "New OpDesc failed");
    GELOGE(FAILED, "[New][OpDesc] failed");
    return nullptr;
  }

  GELOGI("Create StreamActive op:%s.", op_desc->GetName().c_str());
  NodePtr active_node = graph->AddNode(op_desc);
  if (active_node == nullptr) {
    REPORT_CALL_ERROR("E19999", "Add node:%s(%s) to graph:%s failed",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][Node] %s(%s) to graph:%s failed",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    return nullptr;
  }

  if (SetSwitchBranchNodeLabel(active_node, name) != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Set switch branch node label:%s to node:%s(%s) failed",
                      name.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][SwitchBranchNodeLabel] %s to node:%s(%s) failed",
           name.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return nullptr;
  }

  return active_node;
}

///
/// @brief Break NextIteration Link & add name to merge attr
/// @param [in] next_node
/// @param [in] merge_node
/// @return Status
///
Status NextIterationPass::BreakNextIteration(const NodePtr &next_node, NodePtr &merge_node) {
  if ((merge_node == nullptr) || (next_node == nullptr)) {
    GELOGE(PARAM_INVALID, "[Check][Param] merge node or next node is nullptr.");
    return PARAM_INVALID;
  }
  for (const auto &in_anchor : merge_node->GetAllInDataAnchors()) {
    OutDataAnchorPtr out_anchor = in_anchor->GetPeerOutAnchor();
    if ((out_anchor == nullptr) || (out_anchor->GetOwnerNode() != next_node)) {
      continue;
    }
    if (GraphUtils::RemoveEdge(out_anchor, in_anchor) != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Remove edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                        out_anchor->GetOwnerNode()->GetName().c_str(), out_anchor->GetOwnerNode()->GetType().c_str(),
                        out_anchor->GetIdx(),
                        merge_node->GetName().c_str(), merge_node->GetType().c_str(), in_anchor->GetIdx());
      GELOGE(INTERNAL_ERROR, "[Remove][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
             out_anchor->GetOwnerNode()->GetName().c_str(), out_anchor->GetOwnerNode()->GetType().c_str(),
             out_anchor->GetIdx(), merge_node->GetName().c_str(), merge_node->GetType().c_str(), in_anchor->GetIdx());
      return INTERNAL_ERROR;
    }
    if (SetNextIteration(merge_node, next_node) != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Set attr NEXT_ITERATION value:%s to node:%s(%s) failed",
                        next_node->GetName().c_str(), merge_node->GetName().c_str(), merge_node->GetType().c_str());
      GELOGE(INTERNAL_ERROR, "[Set][Attr] NEXT_ITERATION value:%s to node:%s(%s) failed",
             next_node->GetName().c_str(), merge_node->GetName().c_str(), merge_node->GetType().c_str());
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

///
/// @brief find target node
/// @param [in] node
/// @param [in] target_type
/// @param [in] is_input
/// @param [out] target_node
/// @return Status
///
Status NextIterationPass::FindTargetNode(const NodePtr &node, const std::string &target_type, bool is_input,
                                         NodePtr &target_node) {
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Check][Param] node is nullptr.");
    return PARAM_INVALID;
  }
  std::vector<NodePtr> nodes;
  if (is_input) {
    for (const auto &tmp_node : node->GetInDataNodes()) {
      nodes.emplace_back(tmp_node);
    }
  } else {
    for (const auto &tmp_node : node->GetOutDataNodes()) {
      nodes.emplace_back(tmp_node);
    }
  }

  for (const auto &tmp_node : nodes) {
    std::string type;
    GE_CHK_STATUS_RET(GetOriginalType(tmp_node, type), "[Get][NodeType] failed.");
    if ((target_type == LOOPCOND) && (type == target_type)) {
      target_node = tmp_node;
      break;
    } else if ((type == target_type) || (type == "Ref" + target_type)) {
      target_node = tmp_node;
      break;
    }
  }

  if ((target_type != SWITCH) && (target_node == nullptr)) {
    REPORT_INNER_ERROR("E19999", "Find target_type:%s node around node:%s(%s) failed",
                       target_type.c_str(), node->GetName().c_str(), node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] Find target_type:%s node around node:%s(%s) failed",
           target_type.c_str(), node->GetName().c_str(), node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

///
/// @brief Clear Status, used for subgraph pass
/// @return SUCCESS
///
Status NextIterationPass::ClearStatus() {
  loop_group_map_.clear();
  return SUCCESS;
}
}  // namespace ge
