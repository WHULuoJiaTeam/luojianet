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
#include "graph/passes/unused_args_clean_pass.h"

#include "graph/utils/node_utils.h"

namespace ge {
Status UnusedArgsCleanPass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  if (graph->GetParentGraph() != nullptr) {
    GELOGD("Subgraph %s skip the UnusedArgsCleanPass", graph->GetName().c_str());
    return SUCCESS;
  }
  GELOGD("Begin to run Unused args clean on graph: %s", graph->GetName().c_str());

  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() != CASE) {
      continue;
    }

    const auto &func_desc = node->GetOpDesc();
    map<ComputeGraphPtr, map<uint32_t, NodePtr>> graph_nodes;
    if (ClassifyDataNodes(graph, func_desc, graph_nodes) != SUCCESS) {
      return FAILED;
    }

    // {subgraph0, {{0, Data}, {1, Data}, {2, Data}, {3, Data}, ..., {n, Data}}}
    // {subgraph1, {{0, Data}, {1, Data}, {2, Data}, {3, Data}, ..., {n, Data}}}
    // {subgraph2, {{0, Data}, {1, Data}, {2, Data}, {3, Data}, ..., {n, Data}}}
    uint32_t unused_args_num = 0;
    uint32_t inputs_args_num = func_desc->GetInputsSize();
    for (size_t i = 1; i < inputs_args_num; ++i) {
      if (UnusedInputTensor(graph_nodes, node, i)) {
        unused_args_num++;
      } else {
        (void)UpdateInputTensor(graph_nodes, node, i, unused_args_num);
      }
    }

    (void)NodeUtils::RemoveInputAnchor(node, inputs_args_num - unused_args_num);
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Create nodes for root graph.
/// @param [in] graph_nodes: Data groups of subgraph.
/// @param [in] func_node: functional Node of Case.
/// @param [in] parent_index: parent index for check.
/// @return true: unused / false: used
///
bool UnusedArgsCleanPass::UnusedInputTensor(const map<ComputeGraphPtr, map<uint32_t, NodePtr>> &graph_nodes,
                                            const NodePtr &func_node, uint32_t parent_index) {
  for (const auto &item : graph_nodes) {
    const auto &nodes = item.second;
    const auto it = nodes.find(parent_index);
    if (it == nodes.end()) {    // not used.
      continue;
    }

    const auto &data = it->second;
    for (const auto out_anchor : data->GetAllOutAnchors()) {
      for (const auto in_anchor : out_anchor->GetPeerAnchors()) {
        if (in_anchor == nullptr) {
          continue;
        }

        return false;
      }
    }
  }

  return RemoveInputTensor(graph_nodes, func_node, parent_index) == SUCCESS;
}

///
/// @ingroup ge
/// @brief Get all Data nodes for all subgraph.
/// @param [in] graph: Root compute graph.
/// @param [in] func_desc: functional OpDesc of Case.
/// @param [out] graph_nodes: Data groups of subgraph.
/// @return 0: SUCCESS / others: FAILED
///
Status UnusedArgsCleanPass::ClassifyDataNodes(const ComputeGraphPtr &graph, const OpDescPtr &func_desc,
                                              map<ComputeGraphPtr, map<uint32_t, NodePtr>> &graph_nodes) {
  for (const auto &name : func_desc->GetSubgraphInstanceNames()) {
    const auto &subgraph = graph->GetSubgraph(name);
    if (subgraph == nullptr) {
      REPORT_CALL_ERROR("E19999", "Get subgraph from graph:%s by name:%s failed",
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
/// @brief Update Case input Tensor.
/// @param [in] graph_nodes: Data groups of subgraph.
/// @param [in] func_node: functional Node of Case.
/// @param [in] parent_index: parent index for update.
/// @param [in] unused_num: unused args num.
/// @return 0: SUCCESS / others: FAILED
///
Status UnusedArgsCleanPass::UpdateInputTensor(const map<ComputeGraphPtr, map<uint32_t, NodePtr>> &graph_nodes,
                                              const NodePtr &func_node, uint32_t parent_index, uint32_t unused_num) {
  if (unused_num == 0) {
    return SUCCESS;
  }

  uint32_t update_index = parent_index - unused_num;
  for (const auto &item : graph_nodes) {
    const auto &nodes = item.second;
    const auto it = nodes.find(parent_index);
    if (it == nodes.end()) {    // not used.
      continue;
    }
    const auto data = it->second;

    if (!AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, update_index)) {
      REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                        data->GetName().c_str(), data->GetType().c_str());
      GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
             data->GetName().c_str(), data->GetType().c_str());
      return FAILED;
    }
  }

  const auto &new_anchor = func_node->GetInDataAnchor(update_index);
  const auto &old_anchor = func_node->GetInDataAnchor(parent_index);
  const auto &out_anchor = old_anchor->GetPeerOutAnchor();
  const auto &out_node = out_anchor->GetOwnerNode();

  const auto &func_desc = func_node->GetOpDesc();
  const auto &old_desc = func_desc->GetInputDesc(parent_index);
  (void)func_desc->UpdateInputDesc(update_index, old_desc);

  GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(out_anchor, new_anchor),
                          "[Add][Edge] between %s(index:%d) and %s(index:%d) failed",
                          out_node->GetName().c_str(), out_anchor->GetIdx(),
                          func_node->GetName().c_str(), update_index);
  GELOGI("Add edge success, func node: %s, node: %s, parent index: %u, update index: %u",
         func_node->GetName().c_str(), out_node->GetName().c_str(), parent_index, update_index);

  GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveEdge(out_anchor, old_anchor),
                          "[Remove][Edge] between %s(index:%d) and %s(index:%d) failed",
                          out_node->GetName().c_str(), out_anchor->GetIdx(),
                          func_node->GetName().c_str(), parent_index);
  GELOGI("Remove edge success, func node: %s, node: %s", func_node->GetName().c_str(), out_node->GetName().c_str());

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Remove Case input Tensor.
/// @param [in] graph_nodes: Data groups of subgraph.
/// @param [in] func_node: functional Node of Case.
/// @param [in] parent_index: parent index for remove.
/// @return 0: SUCCESS / others: FAILED
///
Status UnusedArgsCleanPass::RemoveInputTensor(const map<ComputeGraphPtr, map<uint32_t, NodePtr>> &graph_nodes,
                                              const NodePtr &func_node, uint32_t parent_index) {
  for (const auto &item : graph_nodes) {
    const auto &graph = item.first;
    const auto &nodes = item.second;
    const auto it = nodes.find(parent_index);
    if (it == nodes.end()) {    // not used.
      continue;
    }

    const auto &data = it->second;
    GE_CHK_GRAPH_STATUS_RET(graph->RemoveNode(data),
                            "[Remove][Node] %s from graph:%s failed",
                            data->GetName().c_str(), graph->GetName().c_str());
    GELOGI("Remove Node: %s %s", graph->GetName().c_str(), data->GetName().c_str());
  }

  const auto &old_anchor = func_node->GetInDataAnchor(parent_index);
  const auto &out_anchor = old_anchor->GetPeerOutAnchor();
  const auto &out_node = out_anchor->GetOwnerNode();

  GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveEdge(out_anchor, old_anchor),
                          "[Remove][Edge] between %s(index:%d) and %s(index:%d) failed",
                          out_node->GetName().c_str(), out_anchor->GetIdx(),
                          func_node->GetName().c_str(), parent_index);
  GELOGI("Remove edge: %s %s", out_node->GetName().c_str(), func_node->GetName().c_str());

  if (out_node->GetInDataNodes().size() == 0 && out_node->GetOutAllNodes().size() == 0) {
    GE_CHK_GRAPH_STATUS_RET(out_node->GetOwnerComputeGraph()->RemoveNode(out_node),
                            "[Remove][Node] %s from graph:%s failed",
                            out_node->GetName().c_str(), out_node->GetOwnerComputeGraph()->GetName().c_str());
  }
  return SUCCESS;
}
}  // namespace ge