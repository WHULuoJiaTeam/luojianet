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

#include "graph/passes/prune_pass.h"
#include <deque>
#include <set>
#include <unordered_set>
#include <vector>
#include "framework/common/debug/log.h"
#include "framework/common/types.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/utils/node_utils.h"

namespace ge {
Status PrunePass::Run(ge::ComputeGraphPtr graph) {
  GELOGD("PrunePass Start, graph is [%s]", graph->GetName().c_str());
  if (graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param graph is nullptr, check invalid");
    GELOGE(GE_GRAPH_ISNULL, "[Check][Param] input compute graph is NULL.");
    return GE_GRAPH_ISNULL;
  }
  std::vector<NodePtr> out_nodes;
  std::unordered_set<NodePtr> nodes;
  for (NodePtr &node_ptr : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node_ptr->GetOpDesc());
    nodes.insert(node_ptr);
    if (node_ptr->GetOpDesc()->GetType() == NETOUTPUT) {
      out_nodes.push_back(node_ptr);
    }
  }
  if (out_nodes.empty()) {
    GELOGW("graph [%s] does not contain NETOUTPUT type node,no return value. Do nothing!", graph->GetName().c_str());
    return ge::SUCCESS;
  }

  std::unordered_set<NodePtr> nodes_seen;
  for (NodePtr &node_ptr : out_nodes) {
    std::deque<NodePtr> queue;
    queue.push_back(node_ptr);
    nodes_seen.insert(node_ptr);
    while (!queue.empty()) {
      NodePtr node = queue.front();
      GE_CHECK_NOTNULL(node->GetOpDesc());
      queue.pop_front();
      for (auto &in_node : node->GetInAllNodes()) {
        if (nodes_seen.insert(in_node).second) {
          queue.push_back(in_node);
        }
      }
    }
  }

  for (auto &node_ptr : nodes) {
    if (nodes_seen.count(node_ptr) != 0) {
      continue;
    }
    if (node_ptr->GetOpDesc()->GetType() == DATA || node_ptr->GetOpDesc()->GetType() == AIPPDATA) {
      Status status = ge::GraphUtils::AddEdge(node_ptr->GetOutControlAnchor(), out_nodes[0]->GetInControlAnchor());
      if (status != ge::SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                          node_ptr->GetName().c_str(), node_ptr->GetType().c_str(),
                          out_nodes[0]->GetName().c_str(), out_nodes[0]->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[add][ControlEdge] failed between DATA node[%s] and NETOUTPUT node[%s]!",
               node_ptr->GetOpDesc()->GetName().c_str(), out_nodes[0]->GetOpDesc()->GetName().c_str());
        return INTERNAL_ERROR;
      }
      GELOGI("[PrunePass] add extra control edge between DATA node[%s] and NETOUTPUT node[%s]!",
             node_ptr->GetOpDesc()->GetName().c_str(), out_nodes[0]->GetOpDesc()->GetName().c_str());
      continue;
    }
    // Remove subgraphs on the node before remove it in graph.
    (void)NodeUtils::RemoveSubgraphsOnNode(node_ptr);
    /// Common function:[RemoveNode] will delete not only input node but its constant input node also will be deleted
    (void)graph->RemoveNode(node_ptr);
    GELOGI("[PrunePass] remove graph node [%s]!", node_ptr->GetOpDesc()->GetName().c_str());
  }
  return ge::SUCCESS;
}
}  // namespace ge
