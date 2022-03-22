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

#include "graph/utils/ffts_graph_utils.h"

#include <queue>

#include "graph/debug/ge_util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"

namespace {
static uint32_t kFftsPlusSubgraphNum = 0U;
const uint32_t kMaxiumRecursionDepth = 10U;
}

namespace ge {
graphStatus FftsGraphUtils::GraphPartition(ComputeGraph &graph, const std::set<NodePtr> &unsupported_nodes) {
  if (unsupported_nodes.empty()) {
    GELOGI("Graph:%s, no node is unsupported, skip clipping", graph.GetName().c_str());
    return SUCCESS;
  }

  const auto &ffts_plus_graph = GetFftsPlusGraph(graph);
  GE_CHECK_NOTNULL(ffts_plus_graph);
  std::unordered_set<NodePtr> nodes_need_clip;
  std::unordered_set<ComputeGraphPtr> graphs_need_split;
  GE_CHK_STATUS_RET(CollectClipNodesAndGraphs(ffts_plus_graph, unsupported_nodes, nodes_need_clip, graphs_need_split),
                    "[Collect][NeedClip] nodes and subgraphs in graph %s failed", ffts_plus_graph->GetName().c_str());
  if (nodes_need_clip.empty() && graphs_need_split.empty()) {
    GELOGI("Graph:%s, no node/subgraph need to be clipped, skip", ffts_plus_graph->GetName().c_str());
    return SUCCESS;
  }
  const auto &parent_node = ffts_plus_graph->GetParentNode();
  GE_CHECK_NOTNULL(parent_node);
  // op_desc of node should not be null
  (void)parent_node->GetOpDesc()->DelAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH);

  (void)graphs_need_split.emplace(ffts_plus_graph);
  for (const auto &subgraph : graphs_need_split) {
    if (IsGraphNeedSplit(subgraph, nodes_need_clip)) {
      std::vector<std::pair<bool, std::set<NodePtr>>> split_nodes;
      GE_CHK_STATUS_RET(SplitNodesWithCheck(subgraph, nodes_need_clip, split_nodes),
                        "[Split][Nodes] failed, graph:%s", subgraph->GetName().c_str());
      GE_CHK_STATUS_RET(SplitSubgraph(subgraph, split_nodes),
                        "[Split][Subgraph] %s failed", subgraph->GetName().c_str());
    } else {
      GE_CHK_STATUS_RET(BuildFftsPlusSubgraphWithAllNodes(subgraph),
                        "[Build][FftsPlusSubgraph] failed, graph:%s", subgraph->GetName().c_str());
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus FftsGraphUtils::CollectClipNodesAndGraphs(const ComputeGraphPtr &graph,
                                                      const std::set<NodePtr> &unsupported_nodes,
                                                      std::unordered_set<NodePtr> &nodes_need_clip,
                                                      std::unordered_set<ComputeGraphPtr> &graphs_need_split) {
  for (const auto &node : graph->GetAllNodes()) {
    if (unsupported_nodes.count(node) == 0U) {
      continue;
    }

    (void)nodes_need_clip.emplace(node);
    ComputeGraphPtr cur_graph = node->GetOwnerComputeGraph();
    while (cur_graph != graph) {
      const auto &parent_node = cur_graph->GetParentNode();
      if (parent_node == nullptr) {
        break;
      }
      (void)nodes_need_clip.emplace(parent_node);
      std::vector<ComputeGraphPtr> subgraphs;
      GE_CHK_STATUS_RET(NodeUtils::GetDirectSubgraphs(parent_node, subgraphs), "[Get][Subgraphs] failed for node %s",
                        parent_node->GetName().c_str());
      for (const auto &subgraph : subgraphs) {
        (void)graphs_need_split.emplace(subgraph);
      }
      cur_graph = cur_graph->GetParentGraph();
    }
  }

  return GRAPH_SUCCESS;
}

bool FftsGraphUtils::IsGraphNeedSplit(const ComputeGraphPtr &graph,
                                      const std::unordered_set<NodePtr> &nodes_need_clip) {
  for (const auto &node : graph->GetDirectNode()) {
    if (nodes_need_clip.count(node) > 0U) {
      return true;
    }
  }
  return false;
}

graphStatus FftsGraphUtils::SplitNodesWithCheck(const ComputeGraphPtr &graph,
                                                const std::unordered_set<NodePtr> &nodes_need_clip,
                                                std::vector<std::pair<bool, std::set<NodePtr>>> &split_nodes) {
  // collect src nodes
  std::set<NodePtr> cur_nodes;
  std::set<NodePtr> next_nodes;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetInAllNodes().empty()) {
      if (nodes_need_clip.count(node) == 0U) {
        (void)cur_nodes.insert(node);
      } else {
        (void)next_nodes.insert(node);
      }
    }
  }
  // non-calc nodes should remain in ori-graph
  std::set<NodePtr> calc_nodes;
  CollectCalcNodeInSubgraph(graph, calc_nodes);
  // split nodes
  bool support_flag = false;
  std::set<NodePtr> visited_nodes;
  while (!(cur_nodes.empty() && next_nodes.empty())) {
    const auto &is_cur_stage = [support_flag, nodes_need_clip](const NodePtr &node_ptr) -> bool {
      return (support_flag == (nodes_need_clip.count(node_ptr) == 0U));
    };
    SplitNodes(calc_nodes, is_cur_stage, visited_nodes, cur_nodes, next_nodes);
    std::set<NodePtr> cur_split_nodes;
    for (const auto &cur_node : cur_nodes) {
      if (calc_nodes.count(cur_node) > 0U) {
        (void)cur_split_nodes.insert(cur_node);
      }
    }
    if (!cur_split_nodes.empty()) {
      split_nodes.emplace_back(support_flag, cur_split_nodes);
    }
    support_flag = !support_flag;
    cur_nodes.clear();
    std::swap(cur_nodes, next_nodes);
  }

  return GRAPH_SUCCESS;
}

void FftsGraphUtils::SplitNodes(const std::set<NodePtr> &calc_nodes,
                                const std::function<bool(const NodePtr &)> &is_cur_stage,
                                std::set<NodePtr> &visited_nodes,
                                std::set<NodePtr> &cur_nodes,
                                std::set<NodePtr> &next_nodes) {
  visited_nodes.insert(cur_nodes.begin(), cur_nodes.end());
  std::queue<NodePtr> nodes;
  for (const auto &node : cur_nodes) {
    nodes.push(node);
  }
  while (!nodes.empty()) {
    const auto &node = nodes.front();
    nodes.pop();
    if (calc_nodes.count(node) > 0U) {
      (void)cur_nodes.insert(node);
    } else {
      // op_desc of node should not be null
      (void)node->GetOpDesc()->DelAttr(ATTR_NAME_THREAD_SCOPE_ID);
    }
    (void)visited_nodes.insert(node);
    for (const auto &out_node : node->GetOutAllNodes()) {
      const auto &in_nodes = out_node->GetInAllNodes();
      const bool all_in_node_seen = !std::any_of(in_nodes.begin(), in_nodes.end(),
                                                 [visited_nodes](const NodePtr &node_ptr) {
        return visited_nodes.count(node_ptr) == 0U;
      });
      if (!all_in_node_seen) {
        continue;
      }
      if (is_cur_stage(out_node)) {
        (void)nodes.push(out_node);
      } else {
        (void)next_nodes.insert(out_node);
      }
    }
  }
}

graphStatus FftsGraphUtils::SplitSubgraph(const ComputeGraphPtr &subgraph,
                                          const std::vector<std::pair<bool, std::set<NodePtr>>> &split_nodes) {
  for (const auto &item : split_nodes) {
    if ((item.first) && (!item.second.empty())) {
      const auto &subgraph_name = "FFTS_Plus_Subgraph_" + std::to_string(kFftsPlusSubgraphNum++);
      const auto &new_subgraph = GraphUtils::BuildSubgraphWithNodes(subgraph, item.second, subgraph_name);
      if (new_subgraph == nullptr) {
        REPORT_CALL_ERROR("E19999", "Build subgraph %s failed", subgraph_name.c_str());
        GELOGE(GRAPH_FAILED, "[Build][Subgraph] %s failed", subgraph_name.c_str());
        return GRAPH_FAILED;
      }
      GE_CHK_STATUS_RET(SetAttrForFftsPlusSubgraph(new_subgraph), "[Set][Attr] failed for ffts+ subgraph");
    } else {
      for (const auto &node : item.second) {
        // op_desc of node should not be null
        (void)node->GetOpDesc()->DelAttr(ATTR_NAME_THREAD_SCOPE_ID);
      }
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus FftsGraphUtils::BuildFftsPlusSubgraphWithAllNodes(const ComputeGraphPtr &subgraph) {
  GE_CHECK_NOTNULL(subgraph);
  std::set<NodePtr> calc_nodes;
  CollectCalcNodeInSubgraph(subgraph, calc_nodes);
  const auto &subgraph_name = "FFTS_Plus_Subgraph_" + std::to_string(kFftsPlusSubgraphNum++);
  const auto &new_subgraph = GraphUtils::BuildSubgraphWithNodes(subgraph, calc_nodes, subgraph_name);
  if (new_subgraph == nullptr) {
    REPORT_CALL_ERROR("E19999", "Build subgraph %s failed", subgraph_name.c_str());
    GELOGE(GRAPH_FAILED, "[Build][Subgraph] %s failed", subgraph_name.c_str());
    return GRAPH_FAILED;
  }
  GE_CHK_STATUS_RET(SetAttrForFftsPlusSubgraph(new_subgraph), "[Set][Attr] failed for ffts+ subgraph");

  return GRAPH_SUCCESS;
}

void FftsGraphUtils::CollectCalcNodeInSubgraph(const ComputeGraphPtr &subgraph, std::set<NodePtr> &calc_nodes) {
  std::set<NodePtr> edge_nodes;
  const std::set<std::string> ctrl_goto_types = { LABELSET, LABELGOTOEX, LABELSWITCHBYINDEX };
  // collect end nodes
  CollectEndNodeInSubgraph(subgraph, ctrl_goto_types, edge_nodes);
  // collect start nodes
  std::queue<NodePtr> start_nodes;
  for (const auto &node : subgraph->GetDirectNode()) {
    if ((node->GetType() == DATA) ||
        ((node->GetInAllNodes().empty()) && (ctrl_goto_types.count(node->GetType()) > 0U))) {
      start_nodes.push(node);
    }
  }
  while (!start_nodes.empty()) {
    const auto &cur_node = start_nodes.front();
    start_nodes.pop();
    (void)edge_nodes.insert(cur_node);
    for (const auto &out_node : cur_node->GetOutAllNodes()) {
      if (ctrl_goto_types.count(out_node->GetType()) > 0U) {
        start_nodes.push(out_node);
      }
    }
  }

  for (const auto &node : subgraph->GetDirectNode()) {
    if (edge_nodes.count(node) == 0U) {
      (void)calc_nodes.insert(node);
    }
  }
}

void FftsGraphUtils::CollectEndNodeInSubgraph(const ComputeGraphPtr &subgraph,
                                              const std::set<std::string> &ctrl_goto_types,
                                              std::set<NodePtr> &edge_nodes) {
  const auto &net_output_node = subgraph->FindFirstNodeMatchType(NETOUTPUT);
  if (net_output_node == nullptr) {
    return;
  }
  std::set<NodePtr> out_nodes;
  for (const auto &in_node :  net_output_node->GetInAllNodes()) {
    for (const auto &out_node : in_node->GetOutAllNodes()) {
      (void)out_nodes.insert(out_node);
    }
  }
  std::queue<NodePtr> end_nodes;
  end_nodes.push(net_output_node);
  for (const auto &out_node : out_nodes) {
    if (ctrl_goto_types.count(out_node->GetType()) > 0U) {
      end_nodes.push(out_node);
    }
  }
  while (!end_nodes.empty()) {
    const auto &cur_node = end_nodes.front();
    end_nodes.pop();
    (void)edge_nodes.insert(cur_node);
    for (const auto &out_node : cur_node->GetOutAllNodes()) {
      end_nodes.push(out_node);
    }
  }
}

ComputeGraphPtr FftsGraphUtils::GetFftsPlusGraph(ComputeGraph &graph) {
  const auto &parent_node = graph.GetParentNode();
  if (parent_node == nullptr) {
    REPORT_INNER_ERROR("E19999", "parent node of graph %s is null", graph.GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] parent node of graph %s is null", graph.GetName().c_str());
    return nullptr;
  }
  std::vector<ComputeGraphPtr> subgraphs;
  if (NodeUtils::GetDirectSubgraphs(parent_node, subgraphs) != GRAPH_SUCCESS) {
    REPORT_INNER_ERROR("E19999", "Get subgraph failed, node:%s", parent_node->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][Subgraph] failed, node:%s", parent_node->GetName().c_str());
    return nullptr;
  }
  if (subgraphs.size() != 1U) {
    REPORT_INNER_ERROR("E19999", "Number of subgraphs in parent_node:%s is %zu, graph:%s",
                       parent_node->GetName().c_str(), subgraphs.size(), graph.GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] Number of subgraphs in parent_node:%s is %zu, graph:%s",
           parent_node->GetName().c_str(), subgraphs.size(), graph.GetName().c_str());
    return nullptr;
  }
  return subgraphs[0U];
}

graphStatus FftsGraphUtils::SetAttrForFftsPlusSubgraph(const ComputeGraphPtr &subgraph) {
  const auto &parent_node = subgraph->GetParentNode();
  if (parent_node == nullptr) {
    REPORT_INNER_ERROR("E19999", "Parent node of subgraph %s is null", subgraph->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] Parent node of subgraph %s is null", subgraph->GetName().c_str());
    return GRAPH_FAILED;
  }
  (void)AttrUtils::SetStr(parent_node->GetOpDesc(), ATTR_NAME_FFTS_PLUS_SUB_GRAPH, subgraph->GetName().c_str());
  for (const auto &node : subgraph->GetAllNodes()) {
    // depend on SGT api, need modify
    (void)AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_THREAD_SCOPE_ID, 0);
  }
  return GRAPH_SUCCESS;
}

graphStatus FftsGraphUtils::GraphPartition(ComputeGraph &graph,
                                           const CalcFunc &calc_func,
                                           const std::vector<uint32_t> &upper_limit) {
  if ((calc_func == nullptr) || upper_limit.empty()) {
    GELOGI("Graph:%s, calculate function or upper_limit is empty, skip graph partition",
           graph.GetName().c_str());
    return SUCCESS;
  }

  const auto &ffts_plus_graph = GetFftsPlusGraph(graph);
  GE_CHECK_NOTNULL(ffts_plus_graph);
  // calculate value per node / graph
  // value of func_node equal to the sum of all node_value in subgraphs
  std::map<NodePtr, std::vector<uint32_t>> node_value;
  std::map<ComputeGraphPtr, std::vector<uint32_t>> graph_value;
  GE_CHK_STATUS_RET(Calculate(ffts_plus_graph, calc_func, node_value, graph_value),
                    "[Calculate][Value] failed for graph %s", ffts_plus_graph->GetName().c_str());
  if (!IsValueValid(ffts_plus_graph, upper_limit, node_value, graph_value)) {
    REPORT_CALL_ERROR("E19999", "Check value invalid");
    GELOGE(GRAPH_FAILED, "[Check][Value] invalid");
    return GRAPH_FAILED;
  }

  // input graph not exceed the limit
  if ((graph_value.count(ffts_plus_graph) > 0U) && (graph_value[ffts_plus_graph] <= upper_limit)) {
    GELOGI("Graph %s not exceed limit, skip graph partition", ffts_plus_graph->GetName().c_str());
    return SUCCESS;
  }
  const auto &parent_node = ffts_plus_graph->GetParentNode();
  GE_CHECK_NOTNULL(parent_node);
  // op_desc of node should not be null
  (void)parent_node->GetOpDesc()->DelAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH);

  GE_CHK_STATUS_RET(PartitionGraphWithLimit(ffts_plus_graph, node_value, graph_value, upper_limit),
                    "[Partition][Graph] failed, graph:%s", ffts_plus_graph->GetName().c_str());

  // only non-Ffts+ subgraph of PARTITIONEDCALL need to be unfolded
  const auto &filter = [](const ComputeGraphPtr &graph_ptr) {
    const auto &parent = graph_ptr->GetParentNode();
    if ((parent == nullptr) || (parent->GetOpDesc() == nullptr)) {
      return false;
    }
    // op_desc of node should not be null
    if ((parent->GetType() != PARTITIONEDCALL) ||
        (parent->GetOpDesc()->GetSubgraphInstanceNames().size() != 1U)) {
      return false;
    }
    return !parent->GetOpDesc()->HasAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH);
  };
  GE_CHK_STATUS_RET(GraphUtils::UnfoldSubgraph(ffts_plus_graph, filter), "[Unfold][Subgraph] failed, graph:%s",
                    ffts_plus_graph->GetName().c_str());

  return GRAPH_SUCCESS;
}

graphStatus FftsGraphUtils::Calculate(const ComputeGraphPtr &graph,
                                      const CalcFunc &calc_func,
                                      std::map<NodePtr, std::vector<uint32_t>> &node_value,
                                      std::map<ComputeGraphPtr, std::vector<uint32_t>> &graph_value,
                                      const uint32_t recursive_depth) {
  if (recursive_depth >= kMaxiumRecursionDepth) {
    REPORT_INNER_ERROR("E19999", "param depth:%u >= %u(allow max subgraphs)", recursive_depth, kMaxiumRecursionDepth);
    GELOGE(GRAPH_FAILED, "[Check][Param]exist too much subgraphs:%u > %u(allow max subgraphs)",
           recursive_depth, kMaxiumRecursionDepth);
    return GRAPH_FAILED;
  }
  GE_CHECK_NOTNULL(graph);
  std::vector<uint32_t> cur_graph_value;
  for (const auto &node : graph->GetDirectNode()) {
    std::vector<uint32_t> cur_node_value;
    if (node->GetOpDesc()->GetSubgraphInstanceNames().empty()) {
      cur_node_value = calc_func(node);
    } else {
      cur_node_value = Calculate(node, calc_func, node_value, graph_value, recursive_depth);
      if (cur_node_value.empty()) {
        REPORT_INNER_ERROR("E19999", "Calculate value for func node %s failed", node->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Calculate][Value] for func node %s failed", node->GetName().c_str());
        return GRAPH_FAILED;
      }
    }
    node_value[node] = cur_node_value;
    if (cur_graph_value.empty()) {
      cur_graph_value = cur_node_value;
    } else if (cur_graph_value.size() != cur_node_value.size()) {
      REPORT_INNER_ERROR("E19999", "Value size not match, value size of graph %s is %zu, "
                                   "value size of node %s is %zu", graph->GetName().c_str(), cur_graph_value.size(),
                         node->GetName().c_str(), cur_node_value.size());
      GELOGE(GRAPH_FAILED, "[Check][Param] Value size not match, value size of graph %s is %zu, "
                           "value size of node %s is %zu", graph->GetName().c_str(), cur_graph_value.size(),
             node->GetName().c_str(), cur_node_value.size());
      return GRAPH_FAILED;
    } else {
      (void) std::transform(cur_graph_value.begin(), cur_graph_value.end(), cur_node_value.begin(),
                            cur_graph_value.begin(), std::plus<uint32_t>());
    }
  }
  graph_value[graph] = cur_graph_value;
  return SUCCESS;
}

std::vector<uint32_t> FftsGraphUtils::Calculate(const NodePtr &node, const CalcFunc &calc_func,
                                                std::map<NodePtr, std::vector<uint32_t>> &node_value,
                                                std::map<ComputeGraphPtr, std::vector<uint32_t>> &graph_value,
                                                const uint32_t recursive_depth) {
  std::vector<ComputeGraphPtr> subgraphs;
  if (NodeUtils::GetDirectSubgraphs(node, subgraphs) != GRAPH_SUCCESS) {
    REPORT_INNER_ERROR("E19999", "Get subgraphs failed");
    GELOGE(GRAPH_FAILED, "[Get][Subgraphs] failed");
    return {};
  }
  std::vector<uint32_t> cur_node_value;
  for (const auto &subgraph : subgraphs) {
    if (graph_value.count(subgraph) == 0U) {
      if (Calculate(subgraph, calc_func, node_value, graph_value, recursive_depth + 1U) != GRAPH_SUCCESS) {
        REPORT_INNER_ERROR("E19999", "Calculate value failed, graph %s, parent_node:%s",
                           subgraph->GetName().c_str(), node->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Calculate][Value] failed, graph %s, parent_node:%s",
               subgraph->GetName().c_str(), node->GetName().c_str());
        return {};
      }
    }
    if (graph_value.find(subgraph) == graph_value.end()) {
      REPORT_INNER_ERROR("E19999", "Find value failed for graph %s", subgraph->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Find][Value] failed for graph %s", subgraph->GetName().c_str());
      return {};
    }
    const auto &subgraph_value = graph_value[subgraph];
    if (cur_node_value.empty()) {
      cur_node_value = subgraph_value;
    } else if (cur_node_value.size() != subgraph_value.size()) {
      REPORT_INNER_ERROR("E19999", "Value size not match, value size of node %s is %zu, value size of subgraph %s "
                                   "is %zu", node->GetName().c_str(), cur_node_value.size(),
                         subgraph->GetName().c_str(), subgraph_value.size());
      GELOGE(GRAPH_FAILED, "[Check][Param] Value size not match, value size of node %s is %zu, "
                           "value size of subgraph %s is %zu", node->GetName().c_str(), cur_node_value.size(),
             subgraph->GetName().c_str(), subgraph_value.size());
      return {};
    } else {
      (void) std::transform(cur_node_value.begin(), cur_node_value.end(),
                            subgraph_value.begin(), cur_node_value.begin(), std::plus<uint32_t>());
    }
  }

  return cur_node_value;
}

bool FftsGraphUtils::IsValueValid(const ComputeGraphPtr &graph, const std::vector<uint32_t> &upper_limit,
                                  const std::map<NodePtr, std::vector<uint32_t>> &node_value,
                                  const std::map<ComputeGraphPtr, std::vector<uint32_t>> &graph_value) {
  std::vector<ComputeGraphPtr> subgraphs;
  if (GraphUtils::GetSubgraphsRecursively(graph, subgraphs) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Get subgraphs failed, graph:%s", graph->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][Subgraphs] failed, graph:%s", graph->GetName().c_str());
    return false;
  }
  for (const auto &subgraph : subgraphs) {
    if (graph_value.count(subgraph) == 0U) {
      REPORT_INNER_ERROR("E19999", "Find graph value failed, graph:%s", subgraph->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] Find graph value failed, graph:%s", subgraph->GetName().c_str());
      return false;
    }
    std::set<NodePtr> calc_nodes;
    CollectCalcNodeInSubgraph(subgraph, calc_nodes);
    for (const auto &node : calc_nodes) {
      if (node_value.count(node) == 0U) {
        REPORT_INNER_ERROR("E19999", "Find node value failed, node:%s", node->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Check][Param] Find node value failed, node:%s", node->GetName().c_str());
        return false;
      }
    }
  }

  const auto is_node_value_match = [upper_limit](const std::pair<NodePtr, std::vector<uint32_t>> &pair) {
    return pair.second.size() != upper_limit.size();
  };
  if (std::find_if(node_value.begin(), node_value.end(), is_node_value_match) != node_value.end()) {
    REPORT_INNER_ERROR("E19999", "Node value size not match");
    GELOGE(GRAPH_FAILED, "[Check][Param] Node value size not match");
    return false;
  }

  const auto is_graph_value_match = [upper_limit](const std::pair<ComputeGraphPtr, std::vector<uint32_t>> &pair) {
    return pair.second.size() != upper_limit.size();
  };
  if (std::find_if(graph_value.begin(), graph_value.end(), is_graph_value_match) != graph_value.end()) {
    REPORT_INNER_ERROR("E19999", "Graph value size not match");
    GELOGE(GRAPH_FAILED, "[Check][Param] Graph value size not match");
    return false;
  }

  return true;
}

graphStatus FftsGraphUtils::PartitionGraphWithLimit(const ComputeGraphPtr &graph,
                                                    std::map<NodePtr, std::vector<uint32_t>> &node_value,
                                                    std::map<ComputeGraphPtr, std::vector<uint32_t>> &graph_value,
                                                    const std::vector<uint32_t> &upper_limit,
                                                    const uint32_t recursive_depth) {
  if (recursive_depth >= kMaxiumRecursionDepth) {
    REPORT_INNER_ERROR("E19999", "param depth:%u >= %u(allow max subgraphs)", recursive_depth, kMaxiumRecursionDepth);
    GELOGE(GRAPH_FAILED, "[Check][Param]exist too much subgraphs:%u > %u(allow max subgraphs)",
           recursive_depth, kMaxiumRecursionDepth);
    return GRAPH_FAILED;
  }
  GE_CHECK_NOTNULL(graph);
  std::set<NodePtr> calc_nodes;
  CollectCalcNodeInSubgraph(graph, calc_nodes);
  uint32_t split_level = 0U;
  std::map<uint32_t, std::set<NodePtr>> split_nodes;
  std::vector<NodePtr> exceed_single_node;
  std::vector<uint32_t> cur_value;
  for (const auto &node : graph->GetDirectNode()) {
    if (calc_nodes.count(node) == 0U) {
      // op_desc of node should not be null
      (void)node->GetOpDesc()->DelAttr(ATTR_NAME_THREAD_SCOPE_ID);
      continue;
    }
    std::vector<uint32_t> cur_node_value = node_value[node];
    if (cur_value.empty()) {
      cur_value = cur_node_value;
    } else {
      (void)std::transform(cur_value.begin(), cur_value.end(), cur_node_value.begin(), cur_value.begin(),
                           std::plus<uint32_t>());
    }
    if (cur_value <= upper_limit) {
      (void)split_nodes[split_level].emplace(node);
    } else {
      ++split_level;
      if (cur_node_value > upper_limit) {
        (void)exceed_single_node.emplace_back(node);
        cur_value.clear();
      } else {
        (void)split_nodes[split_level].emplace(node);
        cur_value = cur_node_value;
      }
    }
  }

  for (const auto &item : split_nodes) {
    const auto &subgraph_name = "FFTS_Plus_Subgraph_" + std::to_string(kFftsPlusSubgraphNum++);
    const auto &subgraph = GraphUtils::BuildSubgraphWithNodes(graph, item.second, subgraph_name);
    if (subgraph == nullptr) {
      REPORT_CALL_ERROR("E19999", "Build subgraph %s failed", subgraph_name.c_str());
      GELOGE(GRAPH_FAILED, "[Build][Subgraph] %s failed", subgraph_name.c_str());
      return GRAPH_FAILED;
    }
    GE_CHK_STATUS_RET(SetAttrForFftsPlusSubgraph(subgraph), "[Set][Attr] failed for ffts+ subgraph");
  }

  return SplitFuncNode(exceed_single_node, node_value, graph_value, upper_limit, recursive_depth);
}

graphStatus FftsGraphUtils::SplitFuncNode(const std::vector<NodePtr> exceed_single_node,
                                          std::map<NodePtr, std::vector<uint32_t>> &node_value,
                                          std::map<ComputeGraphPtr, std::vector<uint32_t>> &graph_value,
                                          const std::vector<uint32_t> &upper_limit,
                                          const uint32_t recursive_depth) {
  for (const auto &node : exceed_single_node) {
    // op_desc of node should not be null
    (void)node->GetOpDesc()->DelAttr(ATTR_NAME_THREAD_SCOPE_ID);
    std::vector<ComputeGraphPtr> subgraphs;
    GE_CHK_STATUS_RET(NodeUtils::GetDirectSubgraphs(node, subgraphs), "[Get][Subgraphs] of node %s failed",
                      node->GetName().c_str());
    for (const auto &subgraph : subgraphs) {
      if (graph_value[subgraph] <= upper_limit) {
        GE_CHK_STATUS_RET(BuildFftsPlusSubgraphWithAllNodes(subgraph), "[Build][FftsPlusSubgraph] failed, graph:%s ",
                          subgraph->GetName().c_str());
      } else {
        GE_CHK_STATUS_RET(PartitionGraphWithLimit(subgraph, node_value, graph_value, upper_limit, recursive_depth + 1U),
                          "[Partition][Subgraph] failed, graph:%s ", subgraph->GetName().c_str());
      }
    }
  }
  return GRAPH_SUCCESS;
}
}  // namespace ge
