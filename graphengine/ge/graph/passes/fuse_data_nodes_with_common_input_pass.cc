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

#include "graph/passes/fuse_data_nodes_with_common_input_pass.h"

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <set>
#include "framework/common/ge_inner_error_codes.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/node_utils.h"

using std::map;
using std::vector;
using std::set;
using std::string;

namespace ge {
Status FuseDataNodesWithCommonInputPass::Run(ge::ComputeGraphPtr graph) {
  if (graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param graph is nullptr, check invalid");
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[Check][Param] Compute graph is nullptr.");
    return GE_GRAPH_PARAM_NULLPTR;
  }
  GELOGD("FuseDataNodesWithCommonInputPass in.");
  // key: subgraph, value:--key: peer out anchor to parent node, --value: parent indexes to parent node
  map<ComputeGraphPtr, map<OutDataAnchorPtr, set<uint32_t>>> subgraphs_to_need_fuse_nodes_info;
  if (InitNeedFuseNodesInfo(graph, subgraphs_to_need_fuse_nodes_info) != SUCCESS) {
    GELOGE(FAILED, "[Init][NeedFuseNodesInfo] for graph:%s failed.", graph->GetName().c_str());
    return FAILED;
  }
  return FuseDataNodes(subgraphs_to_need_fuse_nodes_info);
}

Status FuseDataNodesWithCommonInputPass::InitNeedFuseNodesInfo(ComputeGraphPtr &graph,
    map<ComputeGraphPtr, map<OutDataAnchorPtr, set<uint32_t>>> &subgraphs_to_need_fuse_nodes_info) {
  for (const auto &subgraph : graph->GetAllSubgraphs()) {
    GE_CHECK_NOTNULL(subgraph);
    auto parent_node = subgraph->GetParentNode();
    GE_CHECK_NOTNULL(parent_node);
    if (parent_node->GetType() == CASE || parent_node->GetType() == IF) {
      auto &peer_out_anchors_to_parent_indexes = subgraphs_to_need_fuse_nodes_info[subgraph];
      for (const auto &in_data_anchor : parent_node->GetAllInDataAnchors()) {
        GE_CHECK_NOTNULL(in_data_anchor);
        OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
        uint32_t parent_index = static_cast<uint32_t>(in_data_anchor->GetIdx());
        GE_CHECK_NOTNULL(peer_out_anchor);
        peer_out_anchors_to_parent_indexes[peer_out_anchor].insert(parent_index);
        GELOGD("Peer node %s is the %d input of parent node %s in %s.",
               peer_out_anchor->GetOwnerNode()->GetName().c_str(), parent_index, parent_node->GetName().c_str(),
               subgraph->GetName().c_str());
      }
    }
  }
  return SUCCESS;
}

Status FuseDataNodesWithCommonInputPass::FuseDataNodes(
    const map<ComputeGraphPtr, map<OutDataAnchorPtr, set<uint32_t>>> &subgraphs_to_need_fuse_nodes_info) {
  for (const auto &subgraph_to_need_fuse_nodes_info : subgraphs_to_need_fuse_nodes_info) {
    auto subgraph = subgraph_to_need_fuse_nodes_info.first;
    for (const auto &peer_out_anchors_to_parent_indexes : subgraph_to_need_fuse_nodes_info.second) {
      if (peer_out_anchors_to_parent_indexes.second.size() <= 1) {
        continue;
      }
      // key: out anchor, value: data nodes with common input will be fused
      map<OutDataAnchorPtr, vector<NodePtr>> peer_out_anchors_to_need_fuse_nodes;
      for (const auto &node : subgraph->GetDirectNode()) {
        if (node->GetType() != DATA) {
          continue;
        }
        GE_CHECK_NOTNULL(node->GetOpDesc());
        uint32_t parent_index = 0;
        if (AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
          if (peer_out_anchors_to_parent_indexes.second.count(parent_index) > 0) {
            peer_out_anchors_to_need_fuse_nodes[peer_out_anchors_to_parent_indexes.first].emplace_back(node);
          }
        }
      }
      for (const auto &peer_out_anchor_to_need_fuse_nodes : peer_out_anchors_to_need_fuse_nodes) {
        auto need_fuse_data_nodes = peer_out_anchor_to_need_fuse_nodes.second;
        auto first_node = need_fuse_data_nodes.at(0);
        for (size_t i = 1; i < need_fuse_data_nodes.size(); ++i) {
          auto node = need_fuse_data_nodes.at(i);
          GELOGI("Replace redundant data node %s by %s exist in graph: %s.", node->GetName().c_str(),
                 first_node->GetName().c_str(), subgraph->GetName().c_str());
          // the data node which can be fused has none input(both data and control in)
          if (GraphUtils::MoveOutCtrlEdges(node, first_node) != SUCCESS) {
            REPORT_CALL_ERROR("E19999", "Move out control edge from node:%s(%s) to node:%s(%s) failed",
                              node->GetName().c_str(), node->GetType().c_str(),
                              first_node->GetName().c_str(), first_node->GetType().c_str());
            return FAILED;
          }
          if (GraphUtils::ReplaceNodeDataAnchors(first_node, node, {}, {0}) != SUCCESS) {
            REPORT_CALL_ERROR("E19999", "Replace data edge from node:%s(%s) to node:%s(%s) failed",
                              node->GetName().c_str(), node->GetType().c_str(),
                              first_node->GetName().c_str(), first_node->GetType().c_str());
            return FAILED;
          }
          if (GraphUtils::RemoveNodeWithoutRelink(subgraph, node) != SUCCESS) {
            REPORT_CALL_ERROR("E19999", "Remove node:%s(%s) without relink in graph:%s failed",
                              node->GetName().c_str(), node->GetType().c_str(), subgraph->GetName().c_str());
            GELOGE(FAILED, "[Remove][Node] %s(%s) without relink in graph:%s failed",
                   node->GetName().c_str(), node->GetType().c_str(), subgraph->GetName().c_str());
            return FAILED;
          }
        }
      }
    }
  }
  return SUCCESS;
}
}  // namespace ge
