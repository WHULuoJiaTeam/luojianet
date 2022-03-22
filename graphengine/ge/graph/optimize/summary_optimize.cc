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

#include <string>
#include <utility>
#include <vector>

#include "graph/optimize/graph_optimize.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "framework/omg/omg_inner_types.h"

namespace {
const char *const kSummary = "Summary";
const int kMaxMapSize = 10000;
}  // namespace

namespace ge {
Status GraphOptimize::HandleSummaryOp(ComputeGraphPtr &compute_graph) {
  GELOGI("[HandleSummaryOp] HandleSummaryOp start!");
  if (summary_output_indexes_.size() >= kMaxMapSize) {
    REPORT_INNER_ERROR("E19999", "Map size:%zu out of range:%d, check invalid.",
                       summary_output_indexes_.size(), kMaxMapSize);
    GELOGE(FAILED, "[Check][Param] Map size:%zu out of range:%d.", summary_output_indexes_.size(), kMaxMapSize);
    return FAILED;
  }
  if (compute_graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param compute_graph is nullptr, check invalid.");
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[Check][Param] compute_graph is nullptr.");
    return GE_GRAPH_PARAM_NULLPTR;
  }
  if (summary_output_indexes_.find(compute_graph->GetGraphID()) != summary_output_indexes_.end()) {
    return SUCCESS;
  }
  vector<NodePtr> del_nodes;
  vector<NodePtr> front_nodes;
  vector<uint8_t> out_index;
  std::map<string, size_t> summary_output_indexes = {};
  size_t output_index = compute_graph->GetGraphOutNodesInfo().size();
  for (auto &node_ptr : compute_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node_ptr);
    OpDescPtr op = node_ptr->GetOpDesc();
    GE_IF_BOOL_EXEC(op == nullptr, GELOGW("op is nullptr!"); continue);

    if (op->GetType() == kSummary) {
      compute_graph->SetSummaryFlag(true);
      auto in = node_ptr->GetInDataAnchor(0);
      if (in == nullptr) {
        REPORT_INNER_ERROR("E19999", "In data anchor(index:0) of node:%s is nullptr", node_ptr->GetName().c_str());
        GELOGE(GE_GRAPH_PARAM_NULLPTR, "[Get][InDataAnchor] of node:%s is nullptr, index:0",
               node_ptr->GetName().c_str());
        return GE_GRAPH_PARAM_NULLPTR;
      }

      auto peerin = in->GetPeerOutAnchor();
      GE_IF_BOOL_EXEC(peerin == nullptr,
                      REPORT_INNER_ERROR("E19999", "peer out anchor is nullptr, node:%s, in anchor index:0",
                                         node_ptr->GetName().c_str());
                      GELOGE(GE_GRAPH_PARAM_NULLPTR, "[Get][PeerOutAnchor] of node:%s is nullptr, in anchor index:0",
                             node_ptr->GetName().c_str());
                      return GE_GRAPH_PARAM_NULLPTR);

      auto ret = GraphUtils::RemoveEdge(peerin, in);
      if (ret != SUCCESS) {
        return ret;
      }

      auto front_node = peerin->GetOwnerNode();
      front_nodes.emplace_back(front_node);
      auto idx = peerin->GetIdx();
      out_index.emplace_back(idx);
      GELOGI("[GraphOptimize] Summary name: %s, output index: %zu", op->GetName().c_str(), output_index);
      summary_output_indexes.emplace(op->GetName(), output_index);
      output_index += 1;

      del_nodes.emplace_back(node_ptr);
    }
  }
  GE_IF_BOOL_EXEC(!summary_output_indexes.empty(), summary_output_indexes_.insert({compute_graph->GetGraphID(),
                  summary_output_indexes}));

  // add output nodes for summary
  std::vector<std::pair<NodePtr, int32_t>> out_nodes_info;
  for (size_t i = 0; i < front_nodes.size(); i++) {
    out_nodes_info.emplace_back(pair<NodePtr, int32_t>(front_nodes[i], out_index[i]));
  }
  compute_graph->AppendGraphOutNodesInfo(out_nodes_info);

  // delete summary node
  for (auto &node_ptr : del_nodes) {
    auto ret = GraphUtils::RemoveNodeWithoutRelink(compute_graph, node_ptr);
    if (ret != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Call RemoveNodeWithoutRelink failed, node:%s, graph:%s",
                        node_ptr->GetName().c_str(), compute_graph->GetName().c_str());
      GELOGE(ret, "[Call][RemoveNodeWithoutRelink] failed, node:%s, graph:%s",
             node_ptr->GetName().c_str(), compute_graph->GetName().c_str());
      return ret;
    }
    // update Target list
    vector<NodePtr> graph_target = compute_graph->GetGraphTargetNodesInfo();
    auto iter = find(graph_target.begin(), graph_target.end(), node_ptr);
    if (iter != graph_target.end()) {
      GELOGI("Current node %s is as Target, remove it from target vector.", node_ptr->GetName().c_str());
      (void)graph_target.erase(iter);
      compute_graph->SetGraphTargetNodesInfo(graph_target);
    }
  }

  return SUCCESS;
}
}  // namespace ge
