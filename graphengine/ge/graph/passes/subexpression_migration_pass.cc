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
#include "graph/passes/subexpression_migration_pass.h"

#include "graph/utils/node_utils.h"
#include "ge_local_engine/engine/host_cpu_engine.h"
#include "graph/passes/folding_pass.h"

namespace ge {
constexpr uint32_t kDataOutIndex = 0;
constexpr uint32_t kCaseInputBase = 1;
constexpr uint32_t kInvalidParent = 0x7fffffffU;

bool IsSameTensor(ConstGeTensorDescPtr src_tensor, ConstGeTensorDescPtr dst_tensor) {
  if ((src_tensor == nullptr) && (dst_tensor == nullptr)) {
    return true;
  }
  if ((src_tensor == nullptr) || (dst_tensor == nullptr)) {
    return false;
  }

  if ((src_tensor->GetDataType() != dst_tensor->GetDataType()) ||
      (src_tensor->GetFormat() != dst_tensor->GetFormat())) {
    return false;
  }

  const auto src_dims = src_tensor->GetShape().GetDims();
  const auto dst_dims = dst_tensor->GetShape().GetDims();
  if (src_dims != dst_dims) {
    return false;
  }

  const auto src_orig_dims = src_tensor->GetOriginShape().GetDims();
  const auto dst_orig_dims = dst_tensor->GetOriginShape().GetDims();
  if (src_orig_dims != dst_orig_dims) {
    return false;
  }

  return true;
}

bool IsSameOpDesc(const OpDescPtr &src_desc, const OpDescPtr &dst_desc) {
  if ((src_desc == nullptr) && (dst_desc == nullptr)) {
    return true;
  }

  if ((src_desc == nullptr) || (dst_desc == nullptr)) {
    return false;
  }

  if (src_desc->GetType() != dst_desc->GetType()) {
    return false;
  }

  if ((src_desc->GetInputsSize() != dst_desc->GetInputsSize()) ||
      (src_desc->GetOutputsSize() != dst_desc->GetOutputsSize())) {
    return false;
  }

  for (uint32_t i = 0; i < src_desc->GetInputsSize(); ++i) {
    if (!IsSameTensor(src_desc->GetInputDescPtr(i), dst_desc->GetInputDescPtr(i))) {
      return false;
    }
  }

  for (uint32_t i = 0; i < src_desc->GetOutputsSize(); ++i) {
    if (!IsSameTensor(src_desc->GetOutputDescPtr(i), dst_desc->GetOutputDescPtr(i))) {
      return false;
    }
  }

  return true;
}

Status SubexpressionMigrationPass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  if (graph->GetParentGraph() != nullptr) {
    GELOGD("Subgraph %s skip the SubexpressionMigrationPass", graph->GetName().c_str());
    return SUCCESS;
  }
  GELOGD("Begin to run Subexpression Migration on graph: %s", graph->GetName().c_str());

  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() != CASE) {
      continue;
    }

    const auto &func_desc = node->GetOpDesc();
    if (!func_desc->HasAttr(ATTR_NAME_BATCH_NUM)) {
      GELOGD("Not multi-batch, Case: %s", node->GetName().c_str());
      continue;
    }

    do {
      migration_append_ = false;
      map<ComputeGraphPtr, map<uint32_t, NodePtr>> graph_nodes;
      if (ClassifyDataNodes(graph, func_desc, graph_nodes) != SUCCESS) {
        return FAILED;
      }

      if (graph_nodes.empty()) {
        GELOGW("Graph: %s nodes is empty", graph->GetName().c_str());
        break;
      }

      // {subgraph0, {{1, Data}, {2, Data}, {3, Data}, {4, Data}, ..., {n, Data}}}
      // {subgraph1, {{1, Data}, {2, Data}, {3, Data}, {4, Data}, ..., {n, Data}}}
      // {subgraph2, {{1, Data}, {2, Data}, {3, Data}, {4, Data}, ..., {n, Data}}}
      const auto base_nodes = graph_nodes.begin()->second;  // Need copy.
      for (const auto &node_item : base_nodes) {
        if (GraphNodeMigration(graph, node, graph_nodes, node_item.second, node_item.first) != SUCCESS) {
          return FAILED;
        }
      }
    } while (migration_append_);
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get all Data nodes for all subgraph.
/// @param [in] graph: Root compute graph.
/// @param [in] func_desc: functional OpDesc of Case.
/// @param [out] graph_nodes: Data groups of subgraph.
/// @return 0: SUCCESS / others: FAILED
///
Status SubexpressionMigrationPass::ClassifyDataNodes(const ComputeGraphPtr &graph, const OpDescPtr &func_desc,
                                                     map<ComputeGraphPtr, map<uint32_t, NodePtr>> &graph_nodes) {
  for (const auto &name : func_desc->GetSubgraphInstanceNames()) {
    const auto &subgraph = graph->GetSubgraph(name);
    if (subgraph == nullptr) {
      REPORT_INNER_ERROR("E19999", "Get subgraph from graph:%s by name:%s failed",
                         graph->GetName().c_str(), name.c_str());
      GELOGE(GE_GRAPH_EMPTY_SUBGRAPH, "[Get][SubGraph] from graph:%s by name:%s failed",
             graph->GetName().c_str(), name.c_str());
      return GE_GRAPH_EMPTY_SUBGRAPH;
    }

    auto &data_nodes = graph_nodes[subgraph];
    for (auto &data : subgraph->GetDirectNode()) {
      if (data->GetType() != DATA) {
        continue;
      }

      uint32_t parent_index = 0;
      if (!AttrUtils::GetInt(data->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
        REPORT_CALL_ERROR("E19999", "Get Attr:%s from op:%s(%s) failed", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                          data->GetName().c_str(), data->GetType().c_str());
        GELOGE(FAILED, "[Get][Attr] %s from op:%s(%s) failed", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
               data->GetName().c_str(), data->GetType().c_str());
        return FAILED;
      }

      data_nodes[parent_index] = data;
      GELOGD("%s, Parent index: %u, Data: %s", subgraph->GetName().c_str(), parent_index, data->GetName().c_str());
    }
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get all Data nodes for all subgraph.
/// @param [in] node: Node Directly to Data.
/// @param [out] inputs: parent index of Input.
/// @param [out] outputs: parent index of Output.
/// @return true: SUCCESS / false: FAILED
///
bool SubexpressionMigrationPass::GetAssociatedNodes(const NodePtr &node, map<uint32_t, uint32_t> &inputs,
                                                    map<uint32_t, uint32_t> &outputs) {
  for (uint32_t i = 0; i < node->GetAllOutDataAnchorsSize(); ++i) {
    outputs[i] = kInvalidParent;
  }

  uint32_t out_index = 0;
  for (uint32_t i = 0; i < node->GetAllInDataAnchorsSize(); ++i) {
    const auto &in_anchor = node->GetInDataAnchor(i);
    const auto &out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
        inputs[i] = kInvalidParent;
        continue;
    }

    // Has none Data input node, Can not move to parent.
    const auto &owner_node = out_anchor->GetOwnerNode();
    if (owner_node->GetType() != DATA) {
      return false;
    }

    uint32_t parent_index = 0;
    if (!AttrUtils::GetInt(owner_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      return false;
    }

    // Input Data feed other Node, need add new Data.
    inputs[i] = parent_index;
    if ((out_index < outputs.size()) && (owner_node->GetOutDataNodesSize() == 1)) {
      outputs[out_index] = parent_index;
      ++out_index;
    }
  }

  return true;
}

///
/// @ingroup ge
/// @brief Get all Data nodes for all subgraph.
/// @param [in] graph_nodes: Data groups of subgraph.
/// @param [in] base_node: Data Node for migration.
/// @param [in] node_idx: Parent index of Data node.
/// @param [in] anchor_idx: Anchor index of node.
/// @return true: Same / false: not same
///
bool SubexpressionMigrationPass::IsParallelNodeSame(const map<ComputeGraphPtr, map<uint32_t, NodePtr>> &graph_nodes,
                                                    const NodePtr &base_node, uint32_t node_idx, uint32_t anchor_idx) {
  auto it = graph_nodes.begin();
  for (++it; it != graph_nodes.end(); ++it) {
    const auto &data_nodes = it->second;
    auto data_it = data_nodes.find(node_idx);
    if (data_it == data_nodes.end()) {
      REPORT_INNER_ERROR("E19999", "Find node in data_nodes by index:%u failed", node_idx);
      GELOGE(FAILED, "[Check][Param] Find node in data_nodes by index:%u failed", node_idx);
      return false;
    }

    const auto &work_data = data_it->second;
    const auto &out_anchor = work_data->GetOutDataAnchor(kDataOutIndex);
    const auto &in_anchors = out_anchor->GetPeerInDataAnchors();
    const auto &in_anchor = in_anchors.at(anchor_idx);
    if (in_anchor == nullptr) {
      REPORT_INNER_ERROR("E19999", "Index:%u anchor not exist in out:%u data anchor's peer of node:%s(%s)",
                         node_idx, kDataOutIndex, work_data->GetName().c_str(), work_data->GetType().c_str());
      GELOGE(FAILED, "[Check][Param] Index:%u anchor not exist in out:%u data anchor's peer of node:%s(%s)",
             node_idx, kDataOutIndex, work_data->GetName().c_str(), work_data->GetType().c_str());
      return false;
    }

    const auto &work_node = in_anchor->GetOwnerNode();
    if (work_node == nullptr) {
      REPORT_INNER_ERROR("E19999", "Owner node of anchor is nullptr, check invalid");
      GELOGE(FAILED, "[Check][Param] Owner node of anchor is nullptr");
      return false;
    }

    if (!IsSameOpDesc(base_node->GetOpDesc(), work_node->GetOpDesc())) {
      GELOGI("OpDesc diff: %s %s", base_node->GetName().c_str(), work_node->GetName().c_str());
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
/// @param [in] graph_nodes: Data groups of subgraph.
/// @param [in] data_base: Data Node for migration.
/// @param [in] data_idx: Data groups of subgraph.
/// @return 0: SUCCESS / others: FAILED
///
Status SubexpressionMigrationPass::GraphNodeMigration(const ComputeGraphPtr &graph, const NodePtr &func_node,
                                                      map<ComputeGraphPtr, map<uint32_t, NodePtr>> &graph_nodes,
                                                      const NodePtr &base_data, uint32_t base_idx) {
  bool can_extrapolation = false;
  do {
    can_extrapolation = false;
    const auto out_anchor = base_data->GetOutDataAnchor(kDataOutIndex);
    const auto in_anchors = out_anchor->GetPeerInDataAnchors();
    for (size_t i = 0; i < in_anchors.size(); ++i) {
      const auto &in_anchor = in_anchors.at(i);
      const auto &base_node = in_anchor->GetOwnerNode();
      GELOGD("Get Data direct node: %s", base_node->GetName().c_str());
      if (!base_node->GetHostNode() || base_node->GetType() == SWITCH) {
        continue;
      }

      // Get associated Data, if Data feed other nodes, need append new Data.
      map<uint32_t, uint32_t> inputs;
      map<uint32_t, uint32_t> outputs;
      if (!GetAssociatedNodes(base_node, inputs, outputs)) {
        continue;
      }

      if (!IsParallelNodeSame(graph_nodes, base_node, base_idx, i)) {
        continue;
      }

      GELOGI("Move to parent: %s, parent index: %u", base_node->GetName().c_str(), base_idx);
      if (AppendParallelNode(graph_nodes, func_node, outputs) != SUCCESS) {
        return FAILED;
      }

      if (MoveNodeToParent(graph, func_node, graph_nodes, i, inputs, outputs) != SUCCESS) {
        return FAILED;
      }
      can_extrapolation = true;
      break;
    }
  } while (can_extrapolation);

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Append Input Tensor for functional node.
/// @param [in] graph_nodes: Data groups of subgraph.
/// @param [in] func_node: functional Node of Case.
/// @param [in] outputs: Parent index of Node output.
/// @return 0: SUCCESS / others: FAILED
///
Status SubexpressionMigrationPass::AppendParallelNode(map<ComputeGraphPtr, map<uint32_t, NodePtr>> &graph_nodes,
                                                      const NodePtr &func_node, map<uint32_t, uint32_t> &outputs) {
  // If outputs index invalid, add Data and Input Tensor.
  for (auto &item : outputs) {
    if (item.second != kInvalidParent) {
      continue;
    }

    // Add Data to subgraph.
    map<ComputeGraphPtr, uint32_t> append_num;
    for (auto &groups : graph_nodes) {
      const auto &subgraph = groups.first;
      auto &data_nodes = groups.second;

      item.second = func_node->GetAllInDataAnchorsSize() + append_num[subgraph]; // Update to valid parent index.
      std::string data_name = subgraph->GetName() + "_data_" + std::to_string(item.second);

      OpDescBuilder op_builder(data_name, DATA);
      const OpDescPtr op_desc = op_builder.AddInput("x").AddOutput("y").Build();
      if (op_desc == nullptr) {
        REPORT_CALL_ERROR("E19999", "Build op:%s(%s) failed", data_name.c_str(), DATA);
        GELOGE(OUT_OF_MEMORY, "[Build][Op] %s(%s) failed", data_name.c_str(), DATA);
        return OUT_OF_MEMORY;
      }

      uint32_t data_index = item.second - kCaseInputBase;
      if (!AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, data_index)) {
        REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_INDEX.c_str(),
                          op_desc->GetName().c_str(), op_desc->GetType().c_str());
        GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_INDEX.c_str(),
               op_desc->GetName().c_str(), op_desc->GetType().c_str());
        return FAILED;
      }

      if (!AttrUtils::SetInt(op_desc, ATTR_NAME_PARENT_NODE_INDEX, item.second)) {
        REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                          op_desc->GetName().c_str(), op_desc->GetType().c_str());
        GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
               op_desc->GetName().c_str(), op_desc->GetType().c_str());
        return FAILED;
      }

      append_num[subgraph]++;
      data_nodes[item.second] = subgraph->AddNode(op_desc);
      GELOGI("Add Node: %s, parent index: %u", op_desc->GetName().c_str(), item.second);
    }

    // Add InputTensor to functional Node.
    GE_CHK_GRAPH_STATUS_RET(NodeUtils::AppendInputAnchor(func_node, item.second + 1),
                            "[Append][InputAnchor] for node:%s failed", func_node->GetName().c_str());
    migration_append_ = true;
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Delete Node from all subgraph.
/// @param [in] graph_nodes: Data groups of subgraph.
/// @param [in] detach: Node will move to parent.
/// @param [in] outputs: Parent index of Node output.
/// @return 0: SUCCESS / others: FAILED
///
Status SubexpressionMigrationPass::DetachParallelNode(const map<uint32_t, NodePtr> &graph_datas, const NodePtr &detach,
                                                      const map<uint32_t, uint32_t> &outputs) {
  // Break Data and Move node.
  for (const auto &in_anchor : detach->GetAllInDataAnchors()) {
    const auto &out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
        continue;
    }
    GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveEdge(out_anchor, in_anchor),
                            "[Remove][Edge] between %s and %s failed",
                            out_anchor->GetOwnerNode()->GetName().c_str(), detach->GetName().c_str());

    const auto &owner_node = out_anchor->GetOwnerNode();
    GELOGI("Remove Edge: %s %s", owner_node->GetName().c_str(), detach->GetName().c_str());
  }

  // Break Move and follow, Link Data and follow.
  for (uint32_t i = 0; i < detach->GetAllOutDataAnchorsSize(); ++i) {
    auto it_idx = outputs.find(i);
    if (it_idx == outputs.end()) {
      REPORT_INNER_ERROR("E19999", "Node:%s parent index %u not found, check invalid", detach->GetName().c_str(), i);
      GELOGE(FAILED, "[Check][Param] Node:%s parent index %u not found", detach->GetName().c_str(), i);
      return FAILED;
    }

    auto it_data = graph_datas.find(it_idx->second);
    if (it_data == graph_datas.end()) {
      REPORT_INNER_ERROR("E19999", "Node:%s parent index %u not found, check invalid", detach->GetName().c_str(), i);
      GELOGE(FAILED, "[Check][Param] Node:%s parent index %u not found", detach->GetName().c_str(), i);
      return FAILED;
    }

    const auto &data_node = it_data->second;
    const auto &out_anchor = detach->GetOutDataAnchor(i);

    const auto &out_desc = detach->GetOpDesc()->GetOutputDesc(i);
    const auto &data_desc = data_node->GetOpDesc();
    (void)data_desc->UpdateInputDesc(kDataOutIndex, out_desc);    // Set Data Input to new connect Node.
    (void)data_desc->UpdateOutputDesc(kDataOutIndex, out_desc);   // Set Data Output to new connect Node.

    for (const auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
      if (in_anchor == nullptr) {
          continue;
      }
      GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveEdge(out_anchor, in_anchor),
                              "[Remove][Edge] between %s and %s failed",
                              detach->GetName().c_str(), in_anchor->GetOwnerNode()->GetName().c_str());
      const auto &owner_node = in_anchor->GetOwnerNode();
      GELOGI("Remove Edge: %s %s", detach->GetName().c_str(), owner_node->GetName().c_str());

      const auto &data_out_anchor = data_node->GetOutDataAnchor(kDataOutIndex);
      GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(data_out_anchor, in_anchor),
                              "[Add][Edge] between %s and %s failed",
                              data_node->GetName().c_str(), owner_node->GetName().c_str());
      GELOGI("Add Edge: %s %s", data_node->GetName().c_str(), owner_node->GetName().c_str());
    }
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Move Node to Parent Graph.
/// @param [in] graph: Parent compute graph.
/// @param [in] func_node: functional Node of Case.
/// @param [in] attach: Node will move to parent.
/// @param [in] inputs: Parent index of Node input.
/// @param [in] outputs: Parent index of Node output.
/// @return 0: SUCCESS / others: FAILED
///
Status SubexpressionMigrationPass::AttachParallelNode(const ComputeGraphPtr &graph, const NodePtr &func_node,
                                                      const NodePtr &attach, const map<uint32_t, uint32_t> &inputs,
                                                      const map<uint32_t, uint32_t> &outputs) {
  GE_CHECK_NOTNULL(attach);
  for (uint32_t i = 0; i < attach->GetAllInDataAnchorsSize(); ++i) {
    auto it_idx = inputs.find(i);
    if (it_idx == inputs.end()) {
      REPORT_INNER_ERROR("E19999", "Node:%s parent index %u not found, check invalid", attach->GetName().c_str(), i);
      GELOGE(FAILED, "[Check][Param] Node:%s parent index %u not found", attach->GetName().c_str(), i);
      return FAILED;
    }
    if (it_idx->second == kInvalidParent) {   // Not connect, Skip.
      continue;
    }

    const auto &in_anchor = func_node->GetInDataAnchor(it_idx->second);
    const auto &out_anchor = in_anchor->GetPeerOutAnchor();
    GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(out_anchor, attach->GetInDataAnchor(i)),
                            "[Add][Edge] between %s and %s failed",
                            out_anchor->GetOwnerNode()->GetName().c_str(), attach->GetName().c_str());
    const auto &owner_node = out_anchor->GetOwnerNode();
    GELOGI("Add Edge: %s %s", owner_node->GetName().c_str(), attach->GetName().c_str());
  }

  for (uint32_t i = 0; i < attach->GetAllOutDataAnchorsSize(); ++i) {
    auto it_idx = outputs.find(i);
    if (it_idx == outputs.end()) {
      return FAILED;
    }
    if (it_idx->second == kInvalidParent) {   // Not connect, Skip.
      continue;
    }

    const auto &out_desc = attach->GetOpDesc()->GetOutputDesc(i);
    const auto &func_desc = func_node->GetOpDesc();
    (void)func_desc->UpdateInputDesc(it_idx->second, out_desc);    // Set Data Input to new connect Node.

    const auto &in_anchor = func_node->GetInDataAnchor(it_idx->second);
    const auto &out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor != nullptr) {
      GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveEdge(out_anchor, in_anchor),
                              "[Remove][Edge] between %s and %s failed",
                              out_anchor->GetOwnerNode()->GetName().c_str(), func_node->GetName().c_str());
      const auto &owner_node = out_anchor->GetOwnerNode();
      GELOGI("Remove Edge: %s %s", owner_node->GetName().c_str(), func_node->GetName().c_str());
    }
    GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(attach->GetOutDataAnchor(i), in_anchor),
                            "[Add][Edge] between %s and %s failed",
                            attach->GetName().c_str(), func_node->GetName().c_str());
    GELOGI("Add Edge: %s %s", attach->GetName().c_str(), func_node->GetName().c_str());
  }

  (void)graph->AddNode(attach);
  (void)attach->SetOwnerComputeGraph(graph);
  GELOGI("Add Node: %s %s", graph->GetName().c_str(), attach->GetName().c_str());

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Move node to Parent graph.
/// @param [in] graph: Root compute graph.
/// @param [in] func_node: functional Node of Case.
/// @param [in] graph_nodes: Data groups of subgraph.
/// @param [in] anchor_idx: anchor index of move Node.
/// @param [in] inputs: Parent index of Node input.
/// @param [in] outputs: Parent index of Node output.
/// @return 0: SUCCESS / others: FAILED
///
Status SubexpressionMigrationPass::MoveNodeToParent(const ComputeGraphPtr &graph, const NodePtr &func_node,
                                                    const map<ComputeGraphPtr, map<uint32_t, NodePtr>> &graph_nodes,
                                                    uint32_t anchor_idx, const map<uint32_t, uint32_t> &inputs,
                                                    const map<uint32_t, uint32_t> &outputs) {
  if (inputs.empty()) {
    REPORT_INNER_ERROR("E19999", "Param inputs is empty, check invalid");
    GELOGE(FAILED, "[Check][Param] Param inputs is empty");
    return FAILED;
  }

  NodePtr move_node;
  uint32_t base_index = inputs.begin()->second;
  for (auto &groups : graph_nodes) {
    const auto &subgraph = groups.first;
    const auto &subnodes = groups.second;
    auto it = subnodes.find(base_index);
    if (it == subnodes.end()) {
      REPORT_INNER_ERROR("E19999", "Index:%u data node not found in graph:%s, check invalid",
                         base_index, subgraph->GetName().c_str());
      GELOGE(FAILED, "[Check][Param] Index:%u data node not found in graph:%s",
             base_index, subgraph->GetName().c_str());
      return FAILED;
    }

    const auto &base_data = it->second;
    const auto &out_anchor = base_data->GetOutDataAnchor(kDataOutIndex);
    const auto &in_anchors = out_anchor->GetPeerInDataAnchors();
    const auto &in_anchor = in_anchors.at(anchor_idx);
    if (in_anchor == nullptr) {
      REPORT_INNER_ERROR("E19999", "Index:%u anchor not exist in out:%u data anchor's peer of node:%s(%s)",
                         anchor_idx, kDataOutIndex, base_data->GetName().c_str(), base_data->GetType().c_str());
      GELOGE(FAILED, "[Check][Param] Index:%u anchor not exist in out:%u data anchor's peer of node:%s(%s)",
             anchor_idx, kDataOutIndex, base_data->GetName().c_str(), base_data->GetType().c_str());
      return FAILED;
    }

    move_node = in_anchor->GetOwnerNode();
    if (move_node == nullptr) {
      REPORT_INNER_ERROR("E19999", "Owner node of anchor is nullptr, check invalid");
      GELOGE(FAILED, "[Check][Param] Owner node of anchor is nullptr");
      return FAILED;
    }

    if (DetachParallelNode(subnodes, move_node, outputs) != SUCCESS) {
      GELOGE(FAILED, "[Detach][ParallelNode] failed, move_node:%s", move_node->GetName().c_str());
      return FAILED;
    }

    GE_CHK_GRAPH_STATUS_RET(subgraph->RemoveNode(move_node),
                            "[Remove][Node] %s from graph:%s failed",
                            move_node->GetName().c_str(), graph->GetName().c_str());
    GELOGI("Remove Node: %s %s", subgraph->GetName().c_str(), move_node->GetName().c_str());
  }

  if (AttachParallelNode(graph, func_node, move_node, inputs, outputs) != SUCCESS) {
    return FAILED;
  }

  return SUCCESS;
}
}  // namespace ge
