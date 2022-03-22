/**
 * Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "graph/passes/ref_identity_delete_op_pass.h"
#include <map>
#include <stack>
#include "common/transop_util.h"

namespace ge {
Status RefIdentityDeleteOpPass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() != REFIDENTITY) {
      continue;
    }
    int input_index = 0;
    NodePtr ref_node = GetRefNode(node, input_index);
    CHECK_FALSE_EXEC(GetRefNode(node, input_index) != nullptr,
                     REPORT_CALL_ERROR("E19999", "Get Ref node of node:%s(%s) failed",
                                       node->GetName().c_str(), node->GetType().c_str());
                     GELOGE(FAILED, "[Get][RefNode] of node:%s(%s) failed",
                            node->GetName().c_str(), node->GetType().c_str());
                     return FAILED);
    CHECK_FALSE_EXEC(DealNoOutputRef(ref_node, node, input_index, graph) == SUCCESS,
                     GELOGE(FAILED, "[Deal][NoOutputRef] for node:%s failed, index:%d",
                            node->GetName().c_str(), input_index);
                     return FAILED);
  }
  return SUCCESS;
}

NodePtr RefIdentityDeleteOpPass::GetRefNode(const NodePtr &node, int &input_index) {
  OutDataAnchorPtr out_anchor = node->GetOutDataAnchor(0);
  CHECK_FALSE_EXEC(out_anchor != nullptr, return nullptr);
  for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
    CHECK_FALSE_EXEC(peer_in_anchor != nullptr, continue);
    auto peer_node = peer_in_anchor->GetOwnerNode();
    CHECK_FALSE_EXEC(peer_node != nullptr, continue);
    const auto &peer_op_desc = peer_node->GetOpDesc();
    CHECK_FALSE_EXEC(peer_op_desc != nullptr, return nullptr);
    const auto &peer_input_desc = peer_op_desc->GetInputDescPtr(static_cast<uint32_t>(peer_in_anchor->GetIdx()));
    if (!peer_input_desc->GetRefPortIndex().empty()) {
      input_index = peer_in_anchor->GetIdx();
      return peer_node;
    }
  }
  return nullptr;
}

Status RefIdentityDeleteOpPass::DealNoOutputRef(const NodePtr &node, const NodePtr &ref_identity, int input_index,
                                                const ComputeGraphPtr &graph) {
  NodePtr first_node = nullptr;
  NodePtr variable_ref = GetVariableRef(node, ref_identity, first_node);
  if (variable_ref == nullptr) {
    REPORT_CALL_ERROR("E19999", "Get variable ref of node:%s(%s) failed",
                      node->GetName().c_str(), node->GetType().c_str());
    GELOGE(FAILED, "[Get][VariableRef] of node:%s(%s) failed", node->GetName().c_str(), node->GetType().c_str());
    return FAILED;
  }
  if (first_node->GetName() != variable_ref->GetName()) {
    // Remove the control edge between ref node and variable ref
    // Add a control edge between ref node and trans node
    //                 +-----------+                         +-----------+
    //       +---------+RefIdentity|             +-----------+RefIdentity|
    //       |         +-----+-----+             |           +-----+-----+
    //       |               |                   |                 |
    //       |               v                   |                 v
    // +-----v-----+    +----+----+        +-----v-----+      +----+----+
    // | TransNode |    | RefNode |   ==>  | TransNode +<--C--+ RefNode |
    // +-----+-----+    +----+----+        +-----+-----+      +---------+
    //       |               |                   |
    //       v               C                   v
    // +-----+-----+         |             +-----+-----+
    // |VariableRef+<--------+             |VariableRef|
    // +-----------+                       +-----------+
    auto ret = ge::GraphUtils::AddEdge(node->GetOutControlAnchor(), first_node->GetInControlAnchor());
    if (ret != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                        node->GetName().c_str(), node->GetType().c_str(),
                        first_node->GetName().c_str(), first_node->GetType().c_str());
      GELOGE(FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
             node->GetName().c_str(), node->GetType().c_str(),
             first_node->GetName().c_str(), first_node->GetType().c_str());
      return FAILED;
    }
    ret = ge::GraphUtils::RemoveEdge(node->GetOutControlAnchor(), variable_ref->GetInControlAnchor());
    if (ret != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Remove control edge between op:%s(%s) and op:%s(%s) failed",
                        node->GetName().c_str(), node->GetType().c_str(),
                        first_node->GetName().c_str(), first_node->GetType().c_str());
      GELOGE(FAILED, "[Remove][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
             node->GetName().c_str(), node->GetType().c_str(),
             first_node->GetName().c_str(), first_node->GetType().c_str());
      return FAILED;
    }
  } else {
    //                   +-----------+                         +-----------+
    //       +-----------+RefIdentity|             +-----------+RefIdentity|
    //       |           +-----+-----+             |           +-----+-----+
    //       |                 |                   |                 |
    //       |                 v                   |                 v
    // +-----v-----+      +----+----+        +-----v-----+      +----+----+
    // |VariableRef+<--C--+ RefNode |  ==>   |VariableRef+<--C--+ RefNode |
    // +-----+-----+      +----+----+        +-----------+      +----+----+
    //       |                 |                                     |
    //       |                 v                                     v
    //       |             +---+----+                            +---+----+
    //       +-----C------>+        |                            |        |
    //                     +--------+                            +--------+
    auto ret = RemoveUselessControlEdge(node, variable_ref);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "[Remove][UselessControlEdge] between node:%s(%s) and node:%s(%s) failed.",
             node->GetName().c_str(), node->GetType().c_str(),
             variable_ref->GetName().c_str(), variable_ref->GetType().c_str());
      return FAILED;
    }
  }
  // remove ref identity
  if (GraphUtils::IsolateNode(ref_identity, {0}) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Isolate op:%s(%s) failed",
                      ref_identity->GetName().c_str(), ref_identity->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Isolate][Node] %s, type:%s failed", ref_identity->GetName().c_str(),
           variable_ref->GetType().c_str());
    return FAILED;
  }
  if (GraphUtils::RemoveNodeWithoutRelink(graph, ref_identity) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Remove node:%s(%s) without relink in graph:%s failed",
                      ref_identity->GetName().c_str(), ref_identity->GetType().c_str(), graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Remove][Node] %s, type:%s without relink in graph:%s failed",
           ref_identity->GetName().c_str(), ref_identity->GetType().c_str(), graph->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

ge::NodePtr RefIdentityDeleteOpPass::GetVariableRef(const NodePtr &ref, const NodePtr &ref_identity,
                                                    NodePtr &first_node) {
  const auto &ref_identity_out_anchor = ref_identity->GetOutDataAnchor(0);
  if (ref_identity_out_anchor == nullptr) {
    return nullptr;
  }
  for (auto &peer_in_anchor : ref_identity_out_anchor->GetPeerInDataAnchors()) {
    const auto &peer_node = peer_in_anchor->GetOwnerNode();
    if (peer_node == nullptr || peer_node->GetName() == ref->GetName()) {
      continue;
    }
    // DFS to find variable ref node.
    std::stack<NodePtr> nodes_to_check;
    nodes_to_check.push(peer_node);
    GELOGI("[RefIdentityDeleteOpPass]Start to search variable ref node from %s.", peer_node->GetName().c_str());
    NodePtr cur_node = nullptr;
    while (!nodes_to_check.empty()) {
      cur_node = nodes_to_check.top();
      nodes_to_check.pop();
      const auto &type = cur_node->GetType();
      if (type == VARIABLE && CheckControlEdge(ref, cur_node)) {
        // Target variable ref node found.
        GELOGI("[RefIdentityDeleteOpPass]variable ref node[%s] found.", cur_node->GetName().c_str());
        first_node = peer_node;
        return cur_node;
      }

      int data_index = TransOpUtil::GetTransOpDataIndex(type);
      if (data_index < 0) {
        GELOGI("[RefIdentityDeleteOpPass]Find node[%s] that is not trans op[%s], stop to search its output.",
               cur_node->GetName().c_str(), type.c_str());
        continue;
      }
      const auto &cur_out_anchor = cur_node->GetOutDataAnchor(0);
      if (cur_out_anchor == nullptr) {
        GELOGI("[RefIdentityDeleteOpPass]Get out anchor of [%s] failed, stop to search its output.",
               cur_node->GetName().c_str());
        continue;
      }
      for (const auto &cur_peer_in_anchor : cur_out_anchor->GetPeerInDataAnchors()) {
        const auto &cur_peer_node = cur_peer_in_anchor->GetOwnerNode();
        if (cur_peer_node == nullptr) {
          continue;
        }
        nodes_to_check.push(cur_peer_node);
      }
    }
    GELOGI("[RefIdentityDeleteOpPass]Can not find variable ref node from %s.", peer_node->GetName().c_str());
  }
  GELOGI("[RefIdentityDeleteOpPass]Can not find variable ref node, return nullptr.");
  return nullptr;
}

bool RefIdentityDeleteOpPass::CheckControlEdge(const NodePtr &ref, const NodePtr &variable_ref) {
  const auto &control_out_anchor = ref->GetOutControlAnchor();
  if (control_out_anchor == nullptr) {
    return false;
  }
  const string &variable_ref_name = variable_ref->GetName();
  for (const auto &peer_in_control_anchor : control_out_anchor->GetPeerInControlAnchors()) {
    const auto &node = peer_in_control_anchor->GetOwnerNode();
    if (node != nullptr && node->GetName() == variable_ref_name) {
      return true;
    }
  }
  return false;
}

Status RefIdentityDeleteOpPass::RemoveUselessControlEdge(const NodePtr &ref, const NodePtr &variable_ref) {
  map<string, NodePtr> out_nodes_map;
  for (const auto &out_anchor : ref->GetAllOutDataAnchors()) {
    for (const auto &peer_in_anchor : out_anchor->GetPeerAnchors()) {
      const auto &peer_node = peer_in_anchor->GetOwnerNode();
      if (peer_node == nullptr) {
        continue;
      }
      out_nodes_map[peer_node->GetName()] = peer_node;
    }
  }
  const auto &out_control_anchor = variable_ref->GetOutControlAnchor();
  GE_CHECK_NOTNULL(out_control_anchor);
  for (const auto &peer_in_control_anchor : out_control_anchor->GetPeerInControlAnchors()) {
    const auto &peer_node = peer_in_control_anchor->GetOwnerNode();
    if (peer_node == nullptr) {
      continue;
    }
    if (out_nodes_map.find(peer_node->GetName()) != out_nodes_map.end()) {
      auto ret = ge::GraphUtils::RemoveEdge(out_control_anchor, peer_in_control_anchor);
      if (ret != SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Remove control edge between op:%s(%s) and op:%s(%s) failed",
                          variable_ref->GetName().c_str(), variable_ref->GetType().c_str(),
                          peer_node->GetName().c_str(), peer_node->GetType().c_str());
        GELOGE(FAILED, "[Remove][ControlEdge] between variable ref node[%s] and ref node's peer node[%s] failed",
               variable_ref->GetName().c_str(), peer_node->GetName().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}
}  // namespace ge
