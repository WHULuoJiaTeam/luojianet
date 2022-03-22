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

#include "graph/passes/transop_breadth_fusion_pass.h"

#include <set>
#include <string>

#include "framework/common/types.h"
#include "common/transop_util.h"
#include "graph/utils/node_utils.h"

namespace ge {
Status TransOpBreadthFusionPass::Run(ge::ComputeGraphPtr graph) {
  if (graph == nullptr) {
    return SUCCESS;
  }
  // breadth fusion pass requires new topologic
  Status ret_topo = graph->TopologicalSorting();
  if (ret_topo != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Topological sorting for graph:%s failed", graph->GetName().c_str());
    GELOGE(ret_topo, "[Call][TopologicalSorting] for graph:%s failed.", graph->GetName().c_str());
    return ret_topo;
  }

  for (auto const &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    auto ids_to_trans_nodes = GetOutputTransOpNodes(node);
    for (auto const &id_to_trans_nodes : ids_to_trans_nodes) {
      if (id_to_trans_nodes.second.size() > 1) {
        GELOGI(
            "Begin to breath fusion output trans-op-nodes for %s, "
            "trans id %s, trans-op count %zu",
            node->GetName().c_str(), id_to_trans_nodes.first.c_str(), id_to_trans_nodes.second.size());
        graphStatus status = Fusion(id_to_trans_nodes.second, graph);
        if (status != GRAPH_SUCCESS) {
          return FAILED;
        }
      }
    }
  }
  return SUCCESS;
}

std::string TransOpBreadthFusionPass::GetNodeId(const int anchor_index, const NodePtr &node) {
  std::stringstream id;
  bool trans_data_type = false;
  bool trans_format = false;
  bool trans_shape = false;

  GE_IF_BOOL_EXEC(node == nullptr || node->GetOpDesc() == nullptr,
                  REPORT_INNER_ERROR("E19999", "Param node or its op_desc is nullptr, check invalid");
                  GELOGE(FAILED, "[Check][Param] Param node or its op_desc is nullptr"); return "");

  std::set<std::string> trans_shapes = { RESHAPE, EXPANDDIMS, SQUEEZE };
  std::set<std::string> trans_shape_and_format = { TRANSPOSE, TRANSPOSED, EXPANDDIMS };
  if (node->GetType() == CAST) {
    trans_data_type = true;
  } else if (trans_shape_and_format.count(node->GetType()) > 0) {
    trans_format = true;
    trans_shape = true;
  } else if (node->GetType() == TRANSDATA) {
    trans_data_type = true;
    trans_format = true;
    trans_shape = true;
  } else if (trans_shapes.count(node->GetType()) > 0) {
    trans_shape = true;
  } else if (node->GetType() == REFORMAT) {
    trans_format = true;
  }

  id << node->GetType() << '-' << anchor_index;
  // temp solution, we should not care about which stream the trans op on
  std::string stream_label;
  if (AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label)) {
    GELOGD("Get stream label %s for node %s, add it to fusion id", stream_label.c_str(), node->GetName().c_str());
    id << '-' << stream_label;
  }
  for (const auto &in_ctrl_node : node->GetInControlNodes()) {
    //                    c
    // switch-->Identity ---> node
    // the control edge from a identity node can not be removed
    if (in_ctrl_node->GetType() == IDENTITY) {
      id << "-control-in-" << in_ctrl_node->GetName();
    }
  }
  // [Cascade pointer]
  const auto &input_desc = node->GetOpDesc()->MutableInputDesc(0);
  const auto &output_desc = node->GetOpDesc()->MutableOutputDesc(0);
  GE_CHECK_NOTNULL_EXEC(input_desc, return "");
  GE_CHECK_NOTNULL_EXEC(output_desc, return "");
  if (trans_data_type) {
    id << '-';
    id << static_cast<int>(input_desc->GetDataType());
    id << '-';
    id << static_cast<int>(output_desc->GetDataType());
  }
  if (trans_format) {
    id << '-';
    id << static_cast<int>(input_desc->GetFormat());
    id << '-';
    id << static_cast<int>(output_desc->GetFormat());
  }
  if (trans_shape) {
    id << '-';
    id << JoinDims(",", input_desc->GetShape().GetDims());
    id << '-';
    id << JoinDims(",", output_desc->GetShape().GetDims());
  }

  return id.str();
}

/**
 * Get all transform operators in the output of node.
 * @param node
 * @return std::map
 *     key   - transform operator identifer
 *     value - transform operator set
 */
std::map<std::string, std::vector<NodePtr>> TransOpBreadthFusionPass::GetOutputTransOpNodes(const NodePtr &node) {
  auto result = std::map<std::string, std::vector<NodePtr>>();
  if (node == nullptr) {
    return result;
  }
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    if (out_anchor == nullptr) {
      continue;
    }
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      if (peer_in_anchor == nullptr) {
        continue;
      }

      auto peer_node = peer_in_anchor->GetOwnerNode();
      if (peer_node == nullptr) {
        continue;
      }

      if (TransOpUtil::IsTransOp(peer_node) &&
          peer_in_anchor->GetIdx() == TransOpUtil::GetTransOpDataIndex(peer_node)) {
        auto output_node_id = GetNodeId(out_anchor->GetIdx(), peer_node);
        result[output_node_id].push_back(peer_node);
      }
    }
  }
  return result;
}

/**
 * Reserving Transform operators which with smaller topo index,
 * other transform operators's output edges merge to the reserved transform operator.
 * Removed transform operators have no output edges.
 * @param trans_nodes
 * @param graph
 */
graphStatus TransOpBreadthFusionPass::Fusion(const std::vector<NodePtr> &trans_nodes, ComputeGraphPtr &graph) {
  if (trans_nodes.empty()) {
    return GRAPH_FAILED;
  }

  size_t min_index = 0;
  GE_CHECK_NOTNULL(trans_nodes[0]);
  auto op_desc = trans_nodes[0]->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  int64_t min_id = op_desc->GetId();
  size_t vec_size = trans_nodes.size();
  for (size_t i = 1; i < vec_size; i++) {
    GE_CHECK_NOTNULL(trans_nodes[i]);
    op_desc = trans_nodes[i]->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (op_desc->GetId() < min_id) {
      min_index = i;
      min_id = op_desc->GetId();
    }
  }

  NodePtr node_remain = trans_nodes[min_index];
  for (size_t i = 0; i < trans_nodes.size(); ++i) {
    if (min_index == i) {
      continue;
    }
    graphStatus status = NodeUtils::MoveOutputEdges(trans_nodes[i], node_remain);
    if (status != GRAPH_SUCCESS) {
      return status;
    }
    // remove useless trans_node
    status = GraphUtils::IsolateNode(trans_nodes[i], {});
    if (status != GRAPH_SUCCESS) {
      return status;
    }

    status = GraphUtils::RemoveNodeWithoutRelink(graph, trans_nodes[i]);
    if (status != GRAPH_SUCCESS) {
      return status;
    }
    GELOGD("[Breadth fusion] Remove node %s from graph", trans_nodes[i]->GetName().c_str());
  }
  return GRAPH_SUCCESS;
}

std::string TransOpBreadthFusionPass::JoinDims(const std::string &sp, const std::vector<int64_t> &dims) {
  std::stringstream ss;
  bool first = true;
  for (int64_t dim : dims) {
    if (first) {
      first = false;
    } else {
      ss << sp;
    }
    ss << dim;
  }
  return ss.str();
}
}  // namespace ge
