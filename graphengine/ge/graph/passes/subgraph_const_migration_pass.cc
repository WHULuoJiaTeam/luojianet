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
#include "graph/passes/subgraph_const_migration_pass.h"

#include "graph/utils/node_utils.h"
#include "ge_local_engine/engine/host_cpu_engine.h"
#include "graph/passes/folding_pass.h"

namespace ge {
constexpr uint32_t kZeroIndex = 0;
constexpr uint32_t kCaseInputBase = 1;
constexpr uint32_t kInvalidParent = 0x7fffffffU;
const string kMbatchNodeNameMark = "_ascend_mbatch_batch_";

bool IsSameConstNode(const NodePtr &src_node, const NodePtr &dst_node) {
  if ((src_node == nullptr) && (dst_node == nullptr)) {
    return true;
  }

  if ((src_node == nullptr) || (dst_node == nullptr)) {
    return false;
  }

  if (src_node->GetType() != dst_node->GetType()) {
    return false;
  }

  const GeTensorDesc &src_desc = src_node->GetOpDesc()->GetOutputDesc(kZeroIndex);
  const GeTensorDesc &dst_desc = dst_node->GetOpDesc()->GetOutputDesc(kZeroIndex);
  return (src_desc == dst_desc);
}

/***********************************************************************************************************************
                                                                             +-----------+
                                                                             |   Data    |
                                                                             +-----------+
                                                                                   |
                                                                                   |
                                                                             +-----------+
                                                                             |   Cast    |
                                                                             +-----------+
                                                                                   |
                                                                                   |
                                                                             +-----------+ +-----------+ +-----------+
                                                                             | TransData | |   Data    | |   Data    |
                                                                             +-----------+ +-----------+ +-----------+
                                                                                        \        |        /
                                                                                         \       |       /
                                                                                          \      |      /
                                                                                           \     |     /
 +-----------+ +-----------+ +-----------+ +-----------+ +-----------+    +-----------+    +-----------+
 |   Data    | |   Data    | |   Data    | |   Data    | |   Data    |    |   Data    |    |  Conv2D   |
 +-----------+ +-----------+ +-----------+ +-----------+ +-----------+    +-----------+    +-----------+
        \                 \        |        /                  /                |                |         +-----------+
         \                 \       |       /                  /                 |                |         |   Const   |
          \                 \      |      /                  /                  |                |         +-----------+
           \                 \     |     /                  /                   |                |             /
            \                +-----------+                 /                    |          +-----------+      /
             +---------------|   Const   |----------------+                     |          |  Pooling  |-----+
                             +-----------+                                      |          +-----------+
                                   \                                            |               /
                                    \                                           |              /
                                     \                                    +-----------+       /
                                      +-----------------------------------|  Conv2D   |------+
                                                                          +-----------+
                                                                                |
                                                                                |
                                                                          +-----------+
                                                                          |   Node    |
                                                                          +-----------+
***********************************************************************************************************************/
Status SubgraphConstMigrationPass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  if (graph->GetParentGraph() != nullptr) {
    GELOGD("Subgraph %s skip the SubgraphConstMigrationPass", graph->GetName().c_str());
    return SUCCESS;
  }

  GELOGD("Begin to run Subgraph Const Migration on graph: %s", graph->GetName().c_str());
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() != CASE) {
      continue;
    }

    const auto &func_desc = node->GetOpDesc();
    if (!func_desc->HasAttr(ATTR_NAME_BATCH_NUM)) {
      GELOGD("Not multi-batch, Skip Case: %s", node->GetName().c_str());
      continue;
    }

    map<ComputeGraphPtr, map<string, NodePtr>> all_const_nodes;
    map<ComputeGraphPtr, map<uint32_t, NodePtr>> all_data_nodes;
    if (ClassifyGraphNodes(graph, func_desc, all_const_nodes, all_data_nodes) != SUCCESS) {
      return FAILED;
    }

    if (all_const_nodes.empty()) {
      GELOGW("Graph: %s subgraph is empty", graph->GetName().c_str());
      break;
    }

    // {subgraph0, {{key1, Const}, {key2, Const}, {key3, Const}, {key4, Const}, ..., {keyn, Const}}}
    // {subgraph1, {{key1, Const}, {key2, Const}, {key3, Const}, {key4, Const}, ..., {keyn, Const}}}
    // {subgraph2, {{key1, Const}, {key2, Const}, {key3, Const}, {key4, Const}, ..., {keyn, Const}}}
    const auto &const_nodes = all_const_nodes.begin()->second;
    for (const auto &item : const_nodes) {
      if (GraphNodeMigration(graph, node, all_const_nodes, all_data_nodes, item.second, item.first) != SUCCESS) {
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get all Const/Data nodes for all subgraph.
/// @param [in] graph: Root compute graph.
/// @param [in] func_desc: functional OpDesc of Case.
/// @param [out] all_const_nodes: Const groups of subgraph.
/// @param [out] all_data_nodes: Data groups of subgraph.
/// @return 0: SUCCESS / others: FAILED
///
Status SubgraphConstMigrationPass::ClassifyGraphNodes(const ComputeGraphPtr &graph, const OpDescPtr &func_desc,
                                                      map<ComputeGraphPtr, map<string, NodePtr>> &all_const_nodes,
                                                      map<ComputeGraphPtr, map<uint32_t, NodePtr>> &all_data_nodes) {
  for (const auto &name : func_desc->GetSubgraphInstanceNames()) {
    const auto &subgraph = graph->GetSubgraph(name);
    if (subgraph == nullptr) {
      REPORT_INNER_ERROR("E19999", "Get subgraph from graph:%s by name:%s failed",
                         graph->GetName().c_str(), name.c_str());
      GELOGE(GE_GRAPH_EMPTY_SUBGRAPH, "[Get][SubGraph] from graph:%s by name:%s failed",
             graph->GetName().c_str(), name.c_str());
      return GE_GRAPH_EMPTY_SUBGRAPH;
    }

    set<NodePtr> ctrl_only_const_nodes;
    auto &data_nodes = all_data_nodes[subgraph];
    auto &const_nodes = all_const_nodes[subgraph];
    for (auto &node : subgraph->GetDirectNode()) {
      if (node->GetType() == DATA) {
        uint32_t parent_index = kInvalidParent;
        if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
          REPORT_CALL_ERROR("E19999", "Get Attr:%s from op:%s(%s) failed", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                            node->GetName().c_str(), node->GetType().c_str());
          GELOGE(FAILED, "[Get][Attr] %s from op:%s(%s) failed", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                 node->GetName().c_str(), node->GetType().c_str());
          return FAILED;
        }

        data_nodes[parent_index] = node;
        GELOGD("%s, index: %u, Data: %s", subgraph->GetName().c_str(), parent_index, node->GetName().c_str());
      } else if ((node->GetType() == CONSTANT) && (node->GetOutDataAnchor(kZeroIndex) != nullptr)) {
        set<string> peer_name_list;
        GetPeerNameList(node, peer_name_list);
        if (peer_name_list.empty()) {
          GELOGI("%s, Const: %s, no data output", subgraph->GetName().c_str(), node->GetName().c_str());
          const auto in_all_nodes = node->GetInAllNodes();
          if (in_all_nodes.empty() || std::all_of(in_all_nodes.begin(), in_all_nodes.end(),
                                                  [](const NodePtr &n) { return n->GetType() == DATA; })) {
            ctrl_only_const_nodes.insert(node);
          }
          continue;
        }

        string key_of_const;
        for (const string &name : peer_name_list) {
          key_of_const += (key_of_const.empty() ? name : "_" + name);
        }

        const_nodes[key_of_const] = node;
        GELOGD("%s, Const: %s, Key: %s", subgraph->GetName().c_str(), node->GetName().c_str(), key_of_const.c_str());
      }
    }

    for (auto &node : ctrl_only_const_nodes) {
      GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveNodeWithoutRelink(subgraph, node),
                              "[Remove][Node] without relink failed, node:%s, graph:%s",
                              node->GetName().c_str(), subgraph->GetName().c_str());
    }
  }

  return SUCCESS;
}

void SubgraphConstMigrationPass::GetPeerNameList(const NodePtr &node, set<string> &peer_name_list) {
  const auto &out_anchor = node->GetOutDataAnchor(kZeroIndex);
  for (const auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
    const auto &peer_node = in_anchor->GetOwnerNode();
    // Trim subgraph node name prefix.
    string node_full_name = peer_node->GetName();
    size_t pos = node_full_name.find(kMbatchNodeNameMark);
    if (pos == string::npos) {
      GELOGI("Can not find: %s of multi-batch in node: %s", kMbatchNodeNameMark.c_str(), node_full_name.c_str());
      continue;
    }

    string fixed_name = node_full_name.substr(0, pos);
    pos = node_full_name.find("_", pos + kMbatchNodeNameMark.length());
    if (pos != string::npos) {
      fixed_name += node_full_name.substr(pos);
    }

    peer_name_list.insert(fixed_name + ":" + std::to_string(in_anchor->GetIdx()));
  }
}

///
/// @ingroup ge
/// @brief Get parent_index for Const node migration.
/// @param [in] all_data_nodes: Data groups of subgraph.
/// @param [in] const_node: Const node will process.
/// @param [out] parent_index: parent index for replace Data.
/// @return true: SUCCESS / false: FAILED
///
bool SubgraphConstMigrationPass::GetAssociatedNodes(const map<ComputeGraphPtr, map<uint32_t, NodePtr>> &all_data_nodes,
                                                    const NodePtr &const_node, uint32_t &parent_index) {
  for (const auto in_node : const_node->GetInAllNodes()) {
    if (in_node->GetType() != DATA) {
      return false;
    }

    uint32_t node_index = 0;
    if (!AttrUtils::GetInt(in_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, node_index)) {
      return false;
    }

    // Input Data feed other Node, need add new Data.
    if ((parent_index == kInvalidParent) && in_node->GetOutDataNodes().empty()) {
      parent_index = node_index;
    }
  }

  return true;
}

///
/// @ingroup ge
/// @brief Check parallel node is same for all subgraph.
/// @param [in] all_const_nodes: Const groups of subgraph.
/// @param [in] const_node: Const Node for migration.
/// @param [in] node_key: Key of Const node.
/// @return true: Same / false: not same
///
bool SubgraphConstMigrationPass::IsParallelNodeSame(const map<ComputeGraphPtr, map<string, NodePtr>> &all_const_nodes,
                                                    const NodePtr &const_node, const string &node_key) {
  auto it = all_const_nodes.begin();
  for (++it; it != all_const_nodes.end(); ++it) {
    const auto &const_nodes = it->second;
    auto node_it = const_nodes.find(node_key);
    if (node_it == const_nodes.end()) {
      GELOGW("Const node: %s not fount, key: %s", const_node->GetName().c_str(), node_key.c_str());
      return false;
    }

    const auto &work_node = node_it->second;
    if (!IsSameConstNode(const_node, work_node)) {
      GELOGI("Not same: %s %s, key: %s", const_node->GetName().c_str(), work_node->GetName().c_str(), node_key.c_str());
      return false;
    }
  }

  return true;
}

///
/// @ingroup ge
/// @brief Migration subgraph Node to Root
/// @param [in] graph: Root compute graph.
/// @param [in] func_node: functional Node of Case.
/// @param [in] all_const_nodes: Const groups of subgraph.
/// @param [in] all_data_nodes: Data groups of subgraph.
/// @param [in] const_node: Const Node for migration.
/// @param [in] node_key: Key of Const node for migration.
/// @return 0: SUCCESS / others: FAILED
///
Status SubgraphConstMigrationPass::GraphNodeMigration(const ComputeGraphPtr &graph, const NodePtr &func_node,
                                                      const map<ComputeGraphPtr, map<string, NodePtr>> &all_const_nodes,
                                                      map<ComputeGraphPtr, map<uint32_t, NodePtr>> &all_data_nodes,
                                                      const NodePtr &const_node, const string &node_key) {
  if (!IsParallelNodeSame(all_const_nodes, const_node, node_key)) {
    return SUCCESS;
  }

  // Get associated Data, if Data feed other nodes, need append new Data.
  uint32_t parent_index = kInvalidParent;
  if (!GetAssociatedNodes(all_data_nodes, const_node, parent_index)) {
    return SUCCESS;
  }

  GELOGI("Move node: %s, parent index: %u", const_node->GetName().c_str(), parent_index);
  if (AppendParallelNode(func_node, parent_index, all_data_nodes) != SUCCESS) {
    return FAILED;
  }

  if (MoveNodeToParent(graph, func_node, all_const_nodes, all_data_nodes, node_key, parent_index) != SUCCESS) {
    return FAILED;
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Append Input Tensor for functional node.
/// @param [in] func_node: functional Node of Case.
/// @param [in/out] parent_index: Parent index for migration.
/// @param [in/out] all_data_nodes: Data groups of subgraph.
/// @return 0: SUCCESS / others: FAILED
///
Status SubgraphConstMigrationPass::AppendParallelNode(const NodePtr &func_node, uint32_t &parent_index,
                                                      map<ComputeGraphPtr, map<uint32_t, NodePtr>> &all_data_nodes) {
  // If outputs index invalid, add Data and Input Tensor.
  if (parent_index != kInvalidParent) {
    return SUCCESS;
  }

  // Add Data to subgraph.
  parent_index = func_node->GetAllInDataAnchorsSize();  // Update to valid parent index.
  for (auto &item : all_data_nodes) {
    const auto &subgraph = item.first;
    const auto data_name = subgraph->GetName() + "_data_" + std::to_string(parent_index);
    OpDescBuilder op_builder(data_name, DATA);
    const auto op_desc = op_builder.AddInput("x").AddOutput("y").Build();
    if (op_desc == nullptr) {
      REPORT_CALL_ERROR("E19999", "Build op:%s(%s) failed", data_name.c_str(), DATA);
      GELOGE(OUT_OF_MEMORY, "[Build][OpDesc] %s(%s) failed", data_name.c_str(), DATA);
      return OUT_OF_MEMORY;
    }

    uint32_t data_index = parent_index - kCaseInputBase;
    if (!AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, data_index)) {
      REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_INDEX.c_str(),
                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_INDEX.c_str(),
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return FAILED;
    }

    if (!AttrUtils::SetInt(op_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return FAILED;
    }

    item.second[parent_index] = subgraph->AddNode(op_desc);
    GELOGI("Add Node: %s, parent index: %u", op_desc->GetName().c_str(), parent_index);
  }

  // Add InputTensor to functional Node.
  NodeUtils::AppendInputAnchor(func_node, parent_index + 1);
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Delete Node from subgraph.
/// @param [in] graph: subgraph for process.
/// @param [in] const_node: Node will move to parent.
/// @param [in] data_node: Place holder for Const.
/// @return 0: SUCCESS / others: FAILED
///
Status SubgraphConstMigrationPass::DetachParallelNode(const ComputeGraphPtr &graph, const NodePtr &const_node,
                                                      const NodePtr &data_node) {
  // Break Data and Move node.
  const auto &in_anchor = const_node->GetInControlAnchor();
  const auto out_anchors = in_anchor->GetPeerOutControlAnchors();
  for (const auto out_anchor : out_anchors) {
    GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveEdge(out_anchor, in_anchor),
                            "[Remove][Edge] between %s and %s failed", out_anchor->GetOwnerNode()->GetName().c_str(),
                            const_node->GetName().c_str());
    const auto owner_node = out_anchor->GetOwnerNode();
    GELOGI("Remove Edge: %s %s", owner_node->GetName().c_str(), const_node->GetName().c_str());
    if (owner_node->GetInAllNodes().empty() && owner_node->GetOutAllNodes().empty() && owner_node != data_node) {
      GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveNodeWithoutRelink(graph, owner_node),
                              "[Remove][Node] without relink failed, node:%s", owner_node->GetName().c_str());
    }
  }

  const auto &ctrl_anchor = const_node->GetOutControlAnchor();
  const auto ctrl_anchors = ctrl_anchor->GetPeerInControlAnchors();
  for (const auto in_anchor : ctrl_anchors) {
    GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveEdge(ctrl_anchor, in_anchor),
                            "[Remove][ControlEdge] between %s and %s failed",
                            const_node->GetName().c_str(), in_anchor->GetOwnerNode()->GetName().c_str());
    GELOGI("Remove Edge: %s %s", const_node->GetName().c_str(), in_anchor->GetOwnerNode()->GetName().c_str());

    GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(data_node->GetOutControlAnchor(), in_anchor),
                            "[Add][ControlEdge] between %s and %s failed",
                            data_node->GetName().c_str(), in_anchor->GetOwnerNode()->GetName().c_str());
    GELOGI("Add Edge: %s %s", data_node->GetName().c_str(), in_anchor->GetOwnerNode()->GetName().c_str());
  }

  // Break Move and follow, Link Data and follow.
  const auto &out_anchor = const_node->GetOutDataAnchor(kZeroIndex);
  const auto in_anchors = out_anchor->GetPeerInDataAnchors();
  for (const auto in_anchor : in_anchors) {
    GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveEdge(out_anchor, in_anchor),
                            "[Remove][Edge] between %s and %s failed",
                            const_node->GetName().c_str(), in_anchor->GetOwnerNode()->GetName().c_str());
    GELOGI("Remove Edge: %s %s", const_node->GetName().c_str(), in_anchor->GetOwnerNode()->GetName().c_str());

    GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(data_node->GetOutDataAnchor(kZeroIndex), in_anchor),
                            "[Add][Edge] between %s and %s failed",
                            data_node->GetName().c_str(), in_anchor->GetOwnerNode()->GetName().c_str());
    GELOGI("Add Edge: %s %s", data_node->GetName().c_str(), in_anchor->GetOwnerNode()->GetName().c_str());
  }

  // Update Data op DataType.
  const auto &const_desc = const_node->GetOpDesc();
  const auto &tensor_desc = const_desc->GetOutputDesc(kZeroIndex);
  const auto &data_desc = data_node->GetOpDesc();
  (void)data_desc->UpdateInputDesc(kZeroIndex, tensor_desc);    // Set Data Input to new connect Node.
  (void)data_desc->UpdateOutputDesc(kZeroIndex, tensor_desc);   // Set Data Output to new connect Node.

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Move Node to Parent Graph.
/// @param [in] graph: Parent compute graph.
/// @param [in] func_node: functional Node of Case.
/// @param [in] const_node: Node will move to parent.
/// @param [in] parent_index: Parent index of Node input.
/// @return 0: SUCCESS / others: FAILED
///
Status SubgraphConstMigrationPass::AttachParallelNode(const ComputeGraphPtr &graph, const NodePtr &func_node,
                                                      const NodePtr &const_node, uint32_t parent_index) {
  GE_CHECK_NOTNULL(const_node);
  if (parent_index == kInvalidParent) {
    return INTERNAL_ERROR;
  }

  const auto &func_desc = func_node->GetOpDesc();
  const auto &tensor_desc = const_node->GetOpDesc()->GetOutputDesc(kZeroIndex);
  (void)func_desc->UpdateInputDesc(parent_index, tensor_desc);    // Set Data Input to new connect Node.

  const auto &in_anchor = func_node->GetInDataAnchor(parent_index);
  const auto &out_anchor = in_anchor->GetPeerOutAnchor();
  if (out_anchor != nullptr) {  // Break useless old link.
    GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveEdge(out_anchor, in_anchor),
                            "[Remove][Edge] between %s and %s failed",
                            out_anchor->GetOwnerNode()->GetName().c_str(), func_node->GetName().c_str());
    const auto owner_node = out_anchor->GetOwnerNode();
    GELOGI("Remove Edge: %s %s", owner_node->GetName().c_str(), func_node->GetName().c_str());
    if (owner_node->GetInAllNodes().empty() && owner_node->GetOutAllNodes().empty()) {
      GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveNodeWithoutRelink(graph, owner_node),
                              "[Remove][Node] without relink failed, node:%s", owner_node->GetName().c_str());
    }
  }
  GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(const_node->GetOutDataAnchor(kZeroIndex), in_anchor),
                          "[Add][Edge] between %s and %s failed",
                          const_node->GetName().c_str(), func_node->GetName().c_str());
  GELOGI("Add Edge: %s %s, index: %u", const_node->GetName().c_str(), func_node->GetName().c_str(), parent_index);

  (void)graph->AddNode(const_node);
  (void)const_node->SetOwnerComputeGraph(graph);
  GELOGI("Add Node: %s %s", graph->GetName().c_str(), const_node->GetName().c_str());
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Move node to Parent graph.
/// @param [in] graph: Root compute graph.
/// @param [in] func_node: functional Node of Case.
/// @param [in] graph_nodes: Data groups of subgraph.
/// @param [in] index: anchor index of move Node.
/// @param [in] inputs: Parent index of Node input.
/// @param [in] outputs: Parent index of Node output.
/// @return 0: SUCCESS / others: FAILED
///
Status SubgraphConstMigrationPass::MoveNodeToParent(const ComputeGraphPtr &graph, const NodePtr &func_node,
                                                    const map<ComputeGraphPtr, map<string, NodePtr>> &all_const_nodes,
                                                    const map<ComputeGraphPtr, map<uint32_t, NodePtr>> &all_data_nodes,
                                                    const string &node_key, uint32_t parent_index) {
  if (node_key.empty() || parent_index == kInvalidParent) {
    REPORT_INNER_ERROR("E19999", "Param node_key is empty or param parent_index is 0x%X, check invalid",
                       kInvalidParent);
    GELOGE(FAILED, "[Check][Param] Param node_key is empty or param parent_index is 0x%X", kInvalidParent);
    return FAILED;
  }

  NodePtr move_node;
  for (auto &item : all_const_nodes) {
    const auto &subgraph = item.first;
    const auto it_const = item.second.find(node_key);
    if (it_const == item.second.end()) {
      REPORT_INNER_ERROR("E19999", "Const node name:%s not found in graph:%s, check invalid",
                         node_key.c_str(), subgraph->GetName().c_str());
      GELOGE(FAILED, "[Check][Param] Const node name:%s not found in graph:%s",
             node_key.c_str(), subgraph->GetName().c_str());
      return FAILED;
    }
    move_node = it_const->second;

    const auto it_nodes = all_data_nodes.find(subgraph);
    if (it_nodes == all_data_nodes.end()) {
      REPORT_INNER_ERROR("E19999", "Const node name:%s not found in graph:%s, check invalid",
                         node_key.c_str(), subgraph->GetName().c_str());
      GELOGE(FAILED, "[Check][Param] Const node name:%s not found in graph:%s",
             node_key.c_str(), subgraph->GetName().c_str());
      return FAILED;
    }
    const auto it_data = it_nodes->second.find(parent_index);
    if (it_data == it_nodes->second.end()) {
      REPORT_INNER_ERROR("E19999", "Const node name:%s not found in graph:%s, check invalid",
                         node_key.c_str(), subgraph->GetName().c_str());
      GELOGE(FAILED, "[Check][Param] Const node name:%s not found in graph:%s",
             node_key.c_str(), subgraph->GetName().c_str());
      return FAILED;
    }

    if (DetachParallelNode(subgraph, move_node, it_data->second) != SUCCESS) {
      GELOGE(FAILED, "[Detach][ParallelNode] Data:%s not found, index:%u", move_node->GetName().c_str(), parent_index);
      return FAILED;
    }

    GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveNodeWithoutRelink(subgraph, move_node),
                            "[Remove][Node] without relink failed, node:%s, graph:%s",
                            move_node->GetName().c_str(), subgraph->GetName().c_str());
    GELOGI("Remove Node: %s %s", subgraph->GetName().c_str(), move_node->GetName().c_str());
  }

  if (AttachParallelNode(graph, func_node, move_node, parent_index) != SUCCESS) {
    return FAILED;
  }

  return SUCCESS;
}
}  // namespace ge
