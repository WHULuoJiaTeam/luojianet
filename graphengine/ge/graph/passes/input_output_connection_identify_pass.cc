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

#include "graph/passes/input_output_connection_identify_pass.h"

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common/ge/ge_util.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"

using std::map;
using std::string;
using std::vector;

namespace ge {
namespace {
inline bool IsDataOp(const std::string &node_type) {
  return (node_type == DATA_TYPE) || (node_type == AIPP_DATA_TYPE) || (node_type == ANN_DATA_TYPE);
}
}  // namespace

Status InputOutputConnectionIdentifyPass::Run(ComputeGraphPtr graph) {
  if (graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param graph is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Check][Param] Input param graph is nullptr, "
           "skip identification of nodes that connect to input and output.");
    return PARAM_INVALID;
  }

  if (graph->GetParentGraph() != nullptr) {
    GELOGD("Current graph %s is a subgraph, skip identification of nodes that connect to input and output.",
           graph->GetName().c_str());
    return SUCCESS;
  }

  GELOGD("Start to identify nodes that connect to input and output.");
  if (graph->TopologicalSorting() != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Topological Sorting graph:%s failed", graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Call][TopologicalSorting] for graph:%s failed.", graph->GetName().c_str());
    return INTERNAL_ERROR;
  }

  if (GraphUtils::GetRefMapping(graph, symbol_to_anchors_, anchor_to_symbol_) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Get ref mapping from graph:%s failed", graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][RefMapping] for graph:%s failed.", graph->GetName().c_str());
    return INTERNAL_ERROR;
  }

  map<NodePtr, vector<uint32_t>> connect_input_node_idx_map;
  map<NodePtr, vector<uint32_t>> connect_output_node_idx_map;
  Status status = SUCCESS;
  for (const NodePtr &node : graph->GetDirectNode()) {
    // Not only node type Data is determined.
    if (IsDataOp(node->GetType())) {
      GELOGD("Find nodes that connect to root graph input node: %s.", node->GetName().c_str());
      status = ProcessInputNode(node, connect_input_node_idx_map, connect_output_node_idx_map);
      if (status != SUCCESS) {
        GELOGE(status, "[Process][Nodes] that connect to input node:%s failed.", node->GetName().c_str());
        return status;
      }
    }

    if (node->GetType() == NETOUTPUT) {
      GELOGD("Find nodes that connect to root graph output node: %s.", node->GetName().c_str());
      status = ProcessOutputNode(node, connect_input_node_idx_map, connect_output_node_idx_map);
      if (status != SUCCESS) {
        GELOGE(status, "[Process][Nodes] that connect to output node:%s failed.", node->GetName().c_str());
        return status;
      }
    }
  }

  status = SetNodeAttrOfConnectingInputOutput(connect_input_node_idx_map, connect_output_node_idx_map);
  if (status != SUCCESS) {
    GELOGE(status, "[Set][Attr] for nodes that connect to input and output failed.");
    return status;
  }

  GELOGD("Success to identify nodes that connect to input and output.");
  return SUCCESS;
}

Status InputOutputConnectionIdentifyPass::ProcessInputNode(const NodePtr &node,
                                                           map<NodePtr, vector<uint32_t>> &connect_input_node_idx,
                                                           map<NodePtr, vector<uint32_t>> &connect_output_node_idx) {
  GE_CHECK_NOTNULL(node);
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    // The return ptr of GetAllOutDataAnchors is always valid.
    auto anchor_iter = anchor_to_symbol_.find(NodeIndexIO(node, out_data_anchor->GetIdx(), kOut).ToString());
    if (anchor_iter == anchor_to_symbol_.end()) {
      GELOGW("Current node: %s out_data_anchor: %d is invalid, can not find related symbol.", node->GetName().c_str(),
             out_data_anchor->GetIdx());
      continue;
    }

    const string &symbol = anchor_iter->second;
    auto status = UpdateNodeIdxMap(symbol, connect_input_node_idx, connect_output_node_idx);
    if (status != SUCCESS) {
      GELOGE(status, "[Call][UpdateNodeIdxMap] Failed to update node anchor_index map.");
      return status;
    }
  }
  return SUCCESS;
}

Status InputOutputConnectionIdentifyPass::UpdateNodeIdxMap(const string &symbol_string,
                                                           map<NodePtr, vector<uint32_t>> &connect_input_node_idx,
                                                           map<NodePtr, vector<uint32_t>> &connect_output_node_idx) {
  auto symbol_iter = symbol_to_anchors_.find(symbol_string);
  if (symbol_iter == symbol_to_anchors_.end()) {
    REPORT_CALL_ERROR("E19999", "Can't find symbol:%s in symbol_to_anchors map, check invalid",
                      symbol_string.c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Input param symbol string:%s is invalid.", symbol_string.c_str());
    return PARAM_INVALID;
  }
  const auto &node_index_io_list = symbol_iter->second;
  for (const auto &node_index_io : node_index_io_list) {
    if (node_index_io.io_type_ == kOut) {
      // find node that has shared output memory.
      connect_output_node_idx[node_index_io.node_].emplace_back(node_index_io.index_);
    } else {
      // find node that has shared input memory.
      connect_input_node_idx[node_index_io.node_].emplace_back(node_index_io.index_);
    }
  }
  return SUCCESS;
}

Status InputOutputConnectionIdentifyPass::ProcessOutputNode(const NodePtr &node,
                                                            map<NodePtr, vector<uint32_t>> &connect_input_node_idx,
                                                            map<NodePtr, vector<uint32_t>> &connect_output_node_idx) {
  GE_CHECK_NOTNULL(node);
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    // The return ptr of GetAllInDataAnchors is always valid.
    auto anchor_iter = anchor_to_symbol_.find(NodeIndexIO(node, in_data_anchor->GetIdx(), kIn).ToString());
    if (anchor_iter == anchor_to_symbol_.end()) {
      GELOGW("Current node: %s in_data_anchor: %d is invalid, can not find related symbol.", node->GetName().c_str(),
             in_data_anchor->GetIdx());
      continue;
    }

    const string &symbol = anchor_iter->second;
    auto status = UpdateNodeIdxMap(symbol, connect_input_node_idx, connect_output_node_idx);
    if (status != SUCCESS) {
      GELOGE(status, "[Call][UpdateNodeIdxMap] Failed to update node anchor_index map.");
      return status;
    }
  }
  return SUCCESS;
}

Status InputOutputConnectionIdentifyPass::SetNodeAttrOfConnectingInputOutput(
    const map<NodePtr, vector<uint32_t>> &connect_input_node_idx,
    const map<NodePtr, vector<uint32_t>> &connect_output_node_idx) {
  for (const auto &iter : connect_input_node_idx) {
    GE_CHECK_NOTNULL(iter.first);
    if (iter.first->GetOpDesc() != nullptr) {
      if (!AttrUtils::SetListInt(iter.first->GetOpDesc(), ATTR_NAME_NODE_CONNECT_INPUT, iter.second)) {
        REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_NODE_CONNECT_INPUT.c_str(),
                          iter.first->GetName().c_str(), iter.first->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_NODE_CONNECT_INPUT.c_str(),
               iter.first->GetName().c_str(), iter.first->GetType().c_str());
        return INTERNAL_ERROR;
      }
    }
  }

  for (const auto &iter : connect_output_node_idx) {
    GE_CHECK_NOTNULL(iter.first);
    if (iter.first->GetOpDesc() != nullptr) {
      if (!AttrUtils::SetListInt(iter.first->GetOpDesc(), ATTR_NAME_NODE_CONNECT_OUTPUT, iter.second)) {
        REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_NODE_CONNECT_OUTPUT.c_str(),
                          iter.first->GetName().c_str(), iter.first->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_NODE_CONNECT_OUTPUT.c_str(),
               iter.first->GetName().c_str(), iter.first->GetType().c_str());
        return INTERNAL_ERROR;
      }
    }
  }
  return SUCCESS;
}
}  // namespace ge
