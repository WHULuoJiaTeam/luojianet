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

#include "graph/passes/attach_stream_label_pass.h"
#include "external/ge/ge_api_types.h"
#include "common/omg_util.h"

using std::string;

namespace ge {
Status AttachStreamLabelPass::Run(ComputeGraphPtr graph) {
  GELOGD("AttachStreamLabelPass Enter.");

  std::vector<NodePtr> need_label_nodes;
  std::vector<NodePtr> enter_nodes;
  std::map<NodePtr, NodePtr> branch_head_nodes;
  FindNodes(graph, need_label_nodes, enter_nodes, branch_head_nodes);
  for (const auto &node : need_label_nodes) {
    GE_CHK_STATUS_RET(UpdateCondBranch(node, branch_head_nodes), "[Update][CondBranch] failed, start node:%s.",
                      node->GetName().c_str());
  }
  GE_CHK_STATUS_RET(UpdateEnterNode(enter_nodes),
                    "[Update][EnterNode] in graph:%s failed.", graph->GetName().c_str());

  GELOGD("AttachStreamLabelPass Leave.");
  return SUCCESS;
}

///
/// @brief Find StreamSwitch / StreamMerge / Enter node
/// @param [in] graph
/// @param [out] need_label_nodes
/// @param [out] enter_nodes
/// @param [out] branch_head_nodes
/// @return void
///
void AttachStreamLabelPass::FindNodes(const ComputeGraphPtr &graph, std::vector<NodePtr> &need_label_nodes,
                                      std::vector<NodePtr> &enter_nodes,
                                      std::map<NodePtr, NodePtr> &branch_head_nodes) {
  std::vector<NodePtr> stream_switch_nodes;
  for (const NodePtr &node : graph->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    const std::string &type = op_desc->GetType();
    if ((type == STREAMSWITCH) && op_desc->HasAttr(ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG)) {
      stream_switch_nodes.emplace_back(node);
    } else if ((type == STREAMMERGE) && !op_desc->HasAttr(ATTR_NAME_NEXT_ITERATION)) {
      need_label_nodes.emplace_back(node);
    } else if ((type == ENTER) || (type == REFENTER)) {
      enter_nodes.emplace_back(node);
    }
  }

  for (const auto &node : stream_switch_nodes) {
    for (const auto &out_ctrl_node : node->GetOutControlNodes()) {
      GELOGD("branch_head_node %s of stream_switch %s.", out_ctrl_node->GetName().c_str(), node->GetName().c_str());
      branch_head_nodes[out_ctrl_node] = node;
    }
    need_label_nodes.emplace_back(node);
  }
}

///
/// @brief update cond branch
/// @param [in] node
/// @param [in] branch_head_nodes
/// @return Status
///
Status AttachStreamLabelPass::UpdateCondBranch(const NodePtr &node,
                                               const std::map<NodePtr, NodePtr> &branch_head_nodes) {
  std::string stream_label;
  if (AttachFlag(node, stream_label) != SUCCESS) {
    GELOGE(FAILED, "[Attach][Flag] for node %s failed.", node->GetName().c_str());
    return FAILED;
  }

  std::unordered_set<NodePtr> branch_nodes;
  std::unordered_set<NodePtr> visited;
  std::stack<NodePtr> nodes;
  nodes.push(node);
  static const std::set<std::string> end_type_set = {STREAMSWITCH, STREAMMERGE, MERGE};
  while (!nodes.empty()) {
    NodePtr cur_node = nodes.top();
    nodes.pop();
    if (visited.count(cur_node) > 0) {
      continue;
    }

    const std::string &type = cur_node->GetType();
    for (const auto &out_node : cur_node->GetOutAllNodes()) {
      const std::string &out_type = out_node->GetType();
      const auto &iter = branch_head_nodes.find(node);
      bool stop_flag = (end_type_set.count(out_type) > 0) ||
                       ((iter != branch_head_nodes.end()) && (iter->second != node)) ||
                       (((type == ENTER) || (type == REFENTER)) && (out_type != STREAMACTIVE));
      if (!stop_flag) {
        nodes.push(out_node);
        GELOGD("Insert branch node %s.", out_node->GetName().c_str());
        branch_nodes.insert(out_node);
      }
    }
    visited.insert(cur_node);
  }

  for (const NodePtr &tmp_node : branch_nodes) {
    GELOGD("Attach label %s to node: %s.", stream_label.c_str(), tmp_node->GetName().c_str());
    auto status = SetStreamLabel(tmp_node, stream_label);
    if (status != ge::SUCCESS) {
      GELOGE(status, "[Set][StreamLabel] %s to op:%s(%s) failed.",
             stream_label.c_str(), tmp_node->GetName().c_str(), tmp_node->GetType().c_str());
      return status;
    }
  }

  return SUCCESS;
}

///
/// @brief attach flag
/// @param [in] node
/// @param [out] stream_label
/// @return Status
///
Status AttachStreamLabelPass::AttachFlag(const NodePtr &node, std::string &stream_label) {
  const std::string &type = node->GetType();
  if (type == STREAMSWITCH) {
    if (node->GetInDataNodes().empty()) {
      REPORT_INNER_ERROR("E19999", "In data nodes is empty of op:%s(%s), check invalid",
                         node->GetName().c_str(), node->GetType().c_str());
      GELOGE(INTERNAL_ERROR, "[Get][InDataNodes] node %s has no input_data_node.", node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    stream_label = node->GetInDataNodes().at(0)->GetName();
    bool value = false;
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    GE_CHK_BOOL_EXEC(AttrUtils::GetBool(op_desc, ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG, value),
                     REPORT_CALL_ERROR("E19999", "Get Attr:%s of op:%s(%s) failed",
                                       ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG.c_str(),
                                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
                     return FAILED,
                     "[Get][Attr] %s of op:%s(%s) failed", ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG.c_str(),
                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
    stream_label += (value ? "_t" : "_f");
    auto status = SetActiveLabelList(node, {stream_label});
    if (status != ge::SUCCESS) {
      GELOGE(status, "[Set][ActiveLabelList] %s to op:%s(%s) failed.",
             stream_label.c_str(), node->GetName().c_str(), node->GetType().c_str());
      return status;
    }
  } else if (type == STREAMMERGE) {
    stream_label = node->GetName();
    auto status = SetStreamLabel(node, stream_label);
    if (status != ge::SUCCESS) {
      GELOGE(status, "[Set][StreamLabel] %s to op:%s(%s) failed.",
             stream_label.c_str(), node->GetName().c_str(), node->GetType().c_str());
      return status;
    }
  }

  return SUCCESS;
}

///
/// @brief Update stream_label start with enter nodes
/// @param [in] enter_nodes
/// @return Status
///
Status AttachStreamLabelPass::UpdateEnterNode(const std::vector<NodePtr> &enter_nodes) {
  std::unordered_map<NodePtr, std::vector<NodePtr>> enter_active_map;
  for (const auto &enter_node : enter_nodes) {
    for (const auto &out_ctrl_node : enter_node->GetOutControlNodes()) {
      if (out_ctrl_node->GetType() != STREAMACTIVE) {
        continue;
      }
      if (enter_active_map.find(out_ctrl_node) == enter_active_map.end()) {
        enter_active_map[out_ctrl_node] = {enter_node};
      } else {
        enter_active_map[out_ctrl_node].emplace_back(enter_node);
      }
    }
  }

  for (const auto &pair : enter_active_map) {
    if (SetEnterLabel(pair.second, pair.first) != SUCCESS) {
      GELOGE(FAILED, "[Set][EnterLabel] for enter_nodes failed.");
      return FAILED;
    }

    NodePtr active_node = pair.first;
    GE_CHECK_NOTNULL(active_node);
    std::vector<std::string> active_label_list;
    bool get_attr = AttrUtils::GetListStr(active_node->GetOpDesc(), ATTR_NAME_ACTIVE_LABEL_LIST, active_label_list) &&
                    (active_label_list.size() == 1) && !active_label_list[0].empty();
    if (!get_attr) {
      REPORT_CALL_ERROR("E19999", "Get Attr:%s of op:%s(%s) failed",
                        ATTR_NAME_ACTIVE_LABEL_LIST.c_str(),
                        active_node->GetName().c_str(), active_node->GetType().c_str());
      GELOGE(INTERNAL_ERROR, "[Get][Attr] %s of op:%s(%s) failed.", ATTR_NAME_ACTIVE_LABEL_LIST.c_str(),
             active_node->GetName().c_str(), active_node->GetType().c_str());
      return INTERNAL_ERROR;
    }

    std::stack<NodePtr> nodes;
    for (const auto &enter_node : pair.second) {
      nodes.emplace(enter_node);
    }
    if (UpdateLoopBranch(nodes, active_label_list[0]) != SUCCESS) {
      GELOGE(FAILED, "[Update][StreamLabel] %s for loop_branch failed.", active_label_list[0].c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

///
/// @brief Set stream_label for enter_nodes
/// @param [in] enter_nodes
/// @param [in] active_node
/// @return Status
///
Status AttachStreamLabelPass::SetEnterLabel(const std::vector<NodePtr> &enter_nodes, const NodePtr &active_node) {
  std::string stream_label;
  GE_CHECK_NOTNULL(active_node);
  (void)AttrUtils::GetStr(active_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label);
  if (stream_label.empty()) {
    GELOGD("stream_label of enter_active %s is empty.", active_node->GetName().c_str());
    return SUCCESS;
  }

  for (const auto &enter_node : enter_nodes) {
    auto status = SetStreamLabel(enter_node, stream_label);
    if (status != ge::SUCCESS) {
      GELOGE(status, "[Set][StreamLabel] %s to op:%s(%s) failed.",
             stream_label.c_str(), enter_node->GetName().c_str(), enter_node->GetType().c_str());
      return status;
    }
  }
  return SUCCESS;
}

///
/// @brief Update stream_label for loop_branch
/// @param [in] enter_nodes
/// @param [in] stream_label
/// @param [in] batch_label
/// @return Status
///
Status AttachStreamLabelPass::UpdateLoopBranch(const std::stack<NodePtr> &enter_nodes, const string &stream_label) {
  std::stack<NodePtr> nodes(enter_nodes);
  NodePtr cur_node = nullptr;
  while (!nodes.empty()) {
    cur_node = nodes.top();
    nodes.pop();
    for (const NodePtr &out_node : cur_node->GetOutAllNodes()) {
      OpDescPtr out_desc = out_node->GetOpDesc();
      GE_CHECK_NOTNULL(out_desc);
      std::string out_type = out_desc->GetType();
      bool need_skip =
          out_desc->HasAttr(ATTR_NAME_STREAM_LABEL) || (out_type == ENTER) || (out_type == REFENTER) ||
          (((cur_node->GetType() == ENTER) || (cur_node->GetType() == REFENTER)) && (out_type == STREAMACTIVE));
      if (need_skip) {
        continue;
      }
      GELOGD("Attach label %s to node: %s.", stream_label.c_str(), out_node->GetName().c_str());
      auto status = SetStreamLabel(out_node, stream_label);
      if (status != ge::SUCCESS) {
        GELOGE(status, "[Set][StreamLabel] %s to op:%s(%s) failed.",
               stream_label.c_str(), out_node->GetName().c_str(), out_node->GetType().c_str());
        return status;
      }
      nodes.push(out_node);
    }
  }
  return SUCCESS;
}
}  // namespace ge
