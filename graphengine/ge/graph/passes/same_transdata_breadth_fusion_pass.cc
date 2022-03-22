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

#include "graph/passes/same_transdata_breadth_fusion_pass.h"
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "init/gelib.h"

namespace {
const int kNoTransOp = 1;
}  // namespace

namespace ge {
void SameTransdataBreadthFusionPass::GetSubGraphNodesInfo() {
  vector<vector<NodePtr>> before_transdata_nodes(sub_graph_anchors_.size());
  vector<pair<int, InDataAnchorPtr>> all_transdata_nodes;
  for (size_t i = 0; i < sub_graph_anchors_.size(); ++i) {
    auto nodes_anchor = sub_graph_anchors_[i];
    auto iter = nodes_anchor.begin();
    auto first_out_anchor = iter->first;
    GE_CHECK_NOTNULL_JUST_RETURN(first_out_anchor);
    before_transdata_nodes[i].push_back(first_out_anchor->GetOwnerNode());
    GELOGD("index:%zu, node:%s, type:%s", i, first_out_anchor->GetOwnerNode()->GetName().c_str(),
           first_out_anchor->GetOwnerNode()->GetType().c_str());
    while (iter != nodes_anchor.end()) {
      auto in_anchor = iter->second;
      GE_CHECK_NOTNULL_JUST_RETURN(in_anchor);
      auto in_node = in_anchor->GetOwnerNode();
      GELOGD("index:%zu, node:%s, type:%s", i, first_out_anchor->GetOwnerNode()->GetName().c_str(),
             first_out_anchor->GetOwnerNode()->GetType().c_str());
      if (in_node->GetType() == TRANSDATA) {
        all_transdata_nodes.emplace_back(i, in_anchor);
      } else {
        before_transdata_nodes[i].push_back(in_node);
      }
      ++iter;
    }
    GELOGD("index:%zu, before trandata node size:%zu", i, before_transdata_nodes[i].size());
  }
  before_transdata_nodes_.swap(before_transdata_nodes);
  all_transdata_nodes_.swap(all_transdata_nodes);
}

OpDescPtr SameTransdataBreadthFusionPass::GetCastOp(const GeTensorDesc &in_desc, const GeTensorDesc &out_desc) {
  static std::atomic_long atomic_fusion_cast_op_count(1);
  auto fusion_cast_op_count = atomic_fusion_cast_op_count.fetch_add(1);
  std::stringstream cast_op_name;
  cast_op_name << "fusion_cast_" << fusion_cast_op_count;
  auto node_op = ge::OperatorFactory::CreateOperator(cast_op_name.str().c_str(), CAST);
  auto cast_op = ge::OpDescUtils::GetOpDescFromOperator(node_op);
  node_op.BreakConnect();
  if (cast_op == nullptr) {
    REPORT_INNER_ERROR("E19999", "Create Operator:%s(%s) failed", cast_op_name.str().c_str(), CAST);
    GELOGE(INTERNAL_ERROR, "[Get][OpDesc] From Operator:%s(%s) failed", cast_op_name.str().c_str(), CAST);
    return nullptr;
  }
  const int default_output_index = 0;
  const int default_input_index = 0;
  if (cast_op->GetInputsSize() == 0) {
    if (cast_op->AddInputDesc(in_desc) != GRAPH_SUCCESS) {
      GELOGW("AddInputDesc fail.");
    }
  } else {
    if (cast_op->UpdateInputDesc(default_input_index, in_desc) != GRAPH_SUCCESS) {
      GELOGW("UpdateInputDesc fail");
    }
  }

  if (cast_op->GetOutputsSize() == 0) {
    if (cast_op->AddOutputDesc(out_desc) != GRAPH_SUCCESS) {
      GELOGW("AddOutputDesc fail.");
    }
  } else {
    if (cast_op->UpdateOutputDesc(default_output_index, out_desc) != GRAPH_SUCCESS) {
      GELOGW("UpdateOutputDesc fail");
    }
  }
  if (!AttrUtils::SetInt(cast_op, CAST_ATTR_DST_TYPE, static_cast<int64_t>(out_desc.GetDataType()))) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", CAST_ATTR_DST_TYPE.c_str(),
                      cast_op->GetName().c_str(), cast_op->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", CAST_ATTR_DST_TYPE.c_str(),
           cast_op->GetName().c_str(), cast_op->GetType().c_str());
    return nullptr;
  }
  return cast_op;
}

void SameTransdataBreadthFusionPass::InsertSameTransdataNodeIndex(int anchors_index,
                                                                  vector<int> &same_transdata_nodes) {
  auto same_iter = same_transdata_nodes.begin();
  while (same_iter != same_transdata_nodes.end()) {
    if (before_transdata_nodes_[anchors_index].size() <= before_transdata_nodes_[*same_iter].size()) {
      same_transdata_nodes.insert(same_iter, anchors_index);
      return;
    }
    ++same_iter;
  }

  same_transdata_nodes.push_back(anchors_index);
}

std::set<std::string> SameTransdataBreadthFusionPass::GetInControlIdentityNodes(const NodePtr &node,
                                                                                int subgraph_index) {
  std::set<std::string> in_node_names;
  for (const auto &in_node : node->GetInControlNodes()) {
    if (in_node->GetType() == IDENTITY) {
      in_node_names.insert(in_node->GetName());
    }
  }
  for (const auto &subgraph_node : before_transdata_nodes_[subgraph_index]) {
    for (const auto &in_node : subgraph_node->GetInControlNodes()) {
      if (in_node->GetType() == IDENTITY) {
        in_node_names.insert(in_node->GetName());
      }
    }
  }
  GELOGD("control in nodes for %s(%d): %zu", node->GetName().c_str(), subgraph_index, in_node_names.size());
  return in_node_names;
}

void SameTransdataBreadthFusionPass::GetSameTransdataNode(vector<int> &same_transdata_nodes) {
  auto iter = all_transdata_nodes_.begin();
  same_transdata_nodes.push_back(iter->first);

  auto node_for_compare_in_anchor = iter->second;
  GE_CHECK_NOTNULL_JUST_RETURN(node_for_compare_in_anchor);
  auto node_for_compare = node_for_compare_in_anchor->GetOwnerNode();

  // Get op-desc, input/output desc, in-control-edges-from-identity, as the compare-key
  auto op_desc_for_compare = node_for_compare->GetOpDesc();
  GE_CHECK_NOTNULL_JUST_RETURN(op_desc_for_compare);
  string op_compare_stream_label;
  (void)AttrUtils::GetStr(op_desc_for_compare, ATTR_NAME_STREAM_LABEL, op_compare_stream_label);
  auto op_compare_in_ctrl_nodes = GetInControlIdentityNodes(node_for_compare, iter->first);
  auto input_desc_for_compare = op_desc_for_compare->GetInputDescPtr(node_for_compare_in_anchor->GetIdx());
  GE_CHECK_NOTNULL_JUST_RETURN(input_desc_for_compare);
  auto output_desc_for_compare = op_desc_for_compare->GetOutputDescPtr(0);
  GE_CHECK_NOTNULL_JUST_RETURN(output_desc_for_compare);

  iter = all_transdata_nodes_.erase(iter);
  while (iter != all_transdata_nodes_.end()) {
    auto in_anchor = iter->second;
    if (in_anchor == nullptr) {
      continue;
    }
    auto node_tmp = in_anchor->GetOwnerNode();
    if (node_tmp == node_for_compare) {
      ++iter;
      continue;
    }
    GE_CHECK_NOTNULL_JUST_RETURN(node_tmp);
    auto op_desc_tmp = node_tmp->GetOpDesc();
    GE_CHECK_NOTNULL_JUST_RETURN(op_desc_tmp);
    auto input_desc_tmp = op_desc_tmp->GetInputDescPtr(in_anchor->GetIdx());
    auto output_desc_tmp = op_desc_tmp->GetOutputDescPtr(0);
    string op_tmp_stream_label;
    (void)AttrUtils::GetStr(op_desc_tmp, ATTR_NAME_STREAM_LABEL, op_tmp_stream_label);
    auto op_tmp_in_ctrl_nodes = GetInControlIdentityNodes(node_tmp, iter->first);
    GE_CHECK_NOTNULL_JUST_RETURN(input_desc_tmp);
    GE_CHECK_NOTNULL_JUST_RETURN(output_desc_tmp);

    if ((op_compare_stream_label == op_tmp_stream_label) &&
        (input_desc_tmp->GetFormat() == input_desc_for_compare->GetFormat()) &&
        (output_desc_tmp->GetFormat() == output_desc_for_compare->GetFormat()) &&
        (op_compare_in_ctrl_nodes == op_tmp_in_ctrl_nodes)) {
      GELOGD("same transdata node:%s, src node:%s", node_tmp->GetName().c_str(), node_for_compare->GetName().c_str());
      InsertSameTransdataNodeIndex(iter->first, same_transdata_nodes);
      iter = all_transdata_nodes_.erase(iter);
    } else {
      ++iter;
    }
  }
}

graphStatus SameTransdataBreadthFusionPass::ReLinkDataOutput2PreNode(const NodePtr &transdata_node,
                                                                     const OutDataAnchorPtr &pre_out_anchor,
                                                                     const NodePtr &relink_node) {
  GE_CHECK_NOTNULL(pre_out_anchor);
  GE_CHECK_NOTNULL(transdata_node);
  auto transdata_peer_out_control_anchor = pre_out_anchor->GetOwnerNode()->GetOutControlAnchor();
  for (auto &out_anchor : transdata_node->GetAllOutDataAnchors()) {
    // relink data edge
    for (auto &transdata_peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      if (transdata_peer_in_anchor->GetOwnerNode() == relink_node) {
        continue;
      }
      GELOGI("remove edge.src:%s, dst:%s", out_anchor->GetOwnerNode()->GetName().c_str(),
             transdata_peer_in_anchor->GetOwnerNode()->GetName().c_str());
      if (GraphUtils::RemoveEdge(out_anchor, transdata_peer_in_anchor) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Remove edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                          out_anchor->GetOwnerNode()->GetName().c_str(),
                          out_anchor->GetOwnerNode()->GetType().c_str(), out_anchor->GetIdx(),
                          transdata_peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                          transdata_peer_in_anchor->GetOwnerNode()->GetType().c_str(),
                          transdata_peer_in_anchor->GetIdx());
        GELOGE(GRAPH_FAILED, "[Remove][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
               out_anchor->GetOwnerNode()->GetName().c_str(), out_anchor->GetOwnerNode()->GetType().c_str(),
               out_anchor->GetIdx(), transdata_peer_in_anchor->GetOwnerNode()->GetName().c_str(),
               transdata_peer_in_anchor->GetOwnerNode()->GetType().c_str(), transdata_peer_in_anchor->GetIdx());
        return GRAPH_FAILED;
      }
      GELOGI("add edge.src:%s, dst:%s", pre_out_anchor->GetOwnerNode()->GetName().c_str(),
             transdata_peer_in_anchor->GetOwnerNode()->GetName().c_str());
      if (GraphUtils::AddEdge(pre_out_anchor, transdata_peer_in_anchor) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                          pre_out_anchor->GetOwnerNode()->GetName().c_str(),
                          pre_out_anchor->GetOwnerNode()->GetType().c_str(), pre_out_anchor->GetIdx(),
                          transdata_peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                          transdata_peer_in_anchor->GetOwnerNode()->GetType().c_str(),
                          transdata_peer_in_anchor->GetIdx());
        GELOGE(GRAPH_FAILED, "[Add][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
               pre_out_anchor->GetOwnerNode()->GetName().c_str(), pre_out_anchor->GetOwnerNode()->GetType().c_str(),
               pre_out_anchor->GetIdx(), transdata_peer_in_anchor->GetOwnerNode()->GetName().c_str(),
               transdata_peer_in_anchor->GetOwnerNode()->GetType().c_str(), transdata_peer_in_anchor->GetIdx());
        return GRAPH_FAILED;
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::ReLinkOutDataPeerInControlNodes2PreNode(
    const NodePtr &transdata_node, const OutDataAnchorPtr &pre_out_anchor) {
  GE_CHECK_NOTNULL(pre_out_anchor);
  GE_CHECK_NOTNULL(transdata_node);
  auto transdata_peer_out_control_anchor = pre_out_anchor->GetOwnerNode()->GetOutControlAnchor();
  for (auto &out_anchor : transdata_node->GetAllOutDataAnchors()) {
    for (auto &transdata_peer_in_control_anchor : out_anchor->GetPeerInControlAnchors()) {
      GELOGD("remove edge.src:%s, dst:%s", out_anchor->GetOwnerNode()->GetName().c_str(),
             transdata_peer_in_control_anchor->GetOwnerNode()->GetName().c_str());
      if (GraphUtils::RemoveEdge(out_anchor, transdata_peer_in_control_anchor) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Remove control edge between op:%s(%s) and op:%s(%s) failed",
                          out_anchor->GetOwnerNode()->GetName().c_str(),
                          out_anchor->GetOwnerNode()->GetType().c_str(),
                          transdata_peer_in_control_anchor->GetOwnerNode()->GetName().c_str(),
                          transdata_peer_in_control_anchor->GetOwnerNode()->GetType().c_str());
        GELOGE(GRAPH_FAILED, "Remove control edge between op:%s(%s) and op:%s(%s) failed",
               out_anchor->GetOwnerNode()->GetName().c_str(), out_anchor->GetOwnerNode()->GetType().c_str(),
               transdata_peer_in_control_anchor->GetOwnerNode()->GetName().c_str(),
               transdata_peer_in_control_anchor->GetOwnerNode()->GetType().c_str());
        return GRAPH_FAILED;
      }

      if (transdata_peer_out_control_anchor == nullptr) {
        GELOGD("add edge.src:%s, dst:%s", pre_out_anchor->GetOwnerNode()->GetName().c_str(),
               transdata_peer_in_control_anchor->GetOwnerNode()->GetName().c_str());
        if (GraphUtils::AddEdge(pre_out_anchor, transdata_peer_in_control_anchor) != GRAPH_SUCCESS) {
          REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                            pre_out_anchor->GetOwnerNode()->GetName().c_str(),
                            pre_out_anchor->GetOwnerNode()->GetType().c_str(),
                            transdata_peer_in_control_anchor->GetOwnerNode()->GetName().c_str(),
                            transdata_peer_in_control_anchor->GetOwnerNode()->GetType().c_str());
          GELOGE(GRAPH_FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
                 pre_out_anchor->GetOwnerNode()->GetName().c_str(),
                 pre_out_anchor->GetOwnerNode()->GetType().c_str(),
                 transdata_peer_in_control_anchor->GetOwnerNode()->GetName().c_str(),
                 transdata_peer_in_control_anchor->GetOwnerNode()->GetType().c_str());
          return GRAPH_FAILED;
        }
      } else {
        GELOGD("add edge.src node:%s, dst node:%s", pre_out_anchor->GetOwnerNode()->GetName().c_str(),
               transdata_peer_in_control_anchor->GetOwnerNode()->GetName().c_str());
        if (GraphUtils::AddEdge(transdata_peer_out_control_anchor, transdata_peer_in_control_anchor) != GRAPH_SUCCESS) {
          REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                            transdata_peer_out_control_anchor->GetOwnerNode()->GetName().c_str(),
                            transdata_peer_out_control_anchor->GetOwnerNode()->GetType().c_str(),
                            transdata_peer_in_control_anchor->GetOwnerNode()->GetName().c_str(),
                            transdata_peer_in_control_anchor->GetOwnerNode()->GetType().c_str());
          GELOGE(GRAPH_FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
                 transdata_peer_out_control_anchor->GetOwnerNode()->GetName().c_str(),
                 transdata_peer_out_control_anchor->GetOwnerNode()->GetType().c_str(),
                 transdata_peer_in_control_anchor->GetOwnerNode()->GetName().c_str(),
                 transdata_peer_in_control_anchor->GetOwnerNode()->GetType().c_str());
          return GRAPH_FAILED;
        }
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::ReLinkTransdataOutput2PreNode(const NodePtr &transdata_node,
                                                                          const OutDataAnchorPtr &pre_out_anchor,
                                                                          const NodePtr &relink_node) {
  GE_CHECK_NOTNULL(pre_out_anchor);
  if (ReLinkDataOutput2PreNode(transdata_node, pre_out_anchor, relink_node) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  if (ReLinkOutDataPeerInControlNodes2PreNode(transdata_node, pre_out_anchor) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  auto transdata_peer_out_control_anchor = pre_out_anchor->GetOwnerNode()->GetOutControlAnchor();

  return ReLinkTransdataControlOutput2PreNode(transdata_node, pre_out_anchor, transdata_peer_out_control_anchor);
}

graphStatus SameTransdataBreadthFusionPass::ReLinkOutControlPeerInControlAnchors(
    const NodePtr &transdata_node_keep, const OutDataAnchorPtr &pre_out_anchor,
    const OutControlAnchorPtr &transdata_peer_out_control_anchor) {
  GE_CHECK_NOTNULL(transdata_node_keep);
  GE_CHECK_NOTNULL(pre_out_anchor);
  auto out_control_anchor = transdata_node_keep->GetOutControlAnchor();
  if (out_control_anchor == nullptr) {
    return GRAPH_SUCCESS;
  }

  for (auto &transdata_peer_in_control_anchor : out_control_anchor->GetPeerInControlAnchors()) {
    GELOGD("remove edge.src:%s, dst:%s", transdata_node_keep->GetName().c_str(),
           transdata_peer_in_control_anchor->GetOwnerNode()->GetName().c_str());
    if (GraphUtils::RemoveEdge(out_control_anchor, transdata_peer_in_control_anchor) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Remove control edge between op:%s(%s) and op:%s(%s) failed",
                        out_control_anchor->GetOwnerNode()->GetName().c_str(),
                        out_control_anchor->GetOwnerNode()->GetType().c_str(),
                        transdata_peer_in_control_anchor->GetOwnerNode()->GetName().c_str(),
                        transdata_peer_in_control_anchor->GetOwnerNode()->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Remove][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
             out_control_anchor->GetOwnerNode()->GetName().c_str(),
             out_control_anchor->GetOwnerNode()->GetType().c_str(),
             transdata_peer_in_control_anchor->GetOwnerNode()->GetName().c_str(),
             transdata_peer_in_control_anchor->GetOwnerNode()->GetType().c_str());
      return GRAPH_FAILED;
    }

    if (transdata_peer_out_control_anchor == nullptr) {
      GELOGD("add edge.src:%s, dst:%s", pre_out_anchor->GetOwnerNode()->GetName().c_str(),
             transdata_peer_in_control_anchor->GetOwnerNode()->GetName().c_str());
      if (GraphUtils::AddEdge(pre_out_anchor, transdata_peer_in_control_anchor) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                          pre_out_anchor->GetOwnerNode()->GetName().c_str(),
                          pre_out_anchor->GetOwnerNode()->GetType().c_str(),
                          transdata_peer_in_control_anchor->GetOwnerNode()->GetName().c_str(),
                          transdata_peer_in_control_anchor->GetOwnerNode()->GetType().c_str());
        GELOGE(GRAPH_FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
               pre_out_anchor->GetOwnerNode()->GetName().c_str(),
               pre_out_anchor->GetOwnerNode()->GetType().c_str(),
               transdata_peer_in_control_anchor->GetOwnerNode()->GetName().c_str(),
               transdata_peer_in_control_anchor->GetOwnerNode()->GetType().c_str());
        return GRAPH_FAILED;
      }
    } else {
      GELOGD("add edge.src:%s, dst:%s", transdata_peer_out_control_anchor->GetOwnerNode()->GetName().c_str(),
             transdata_peer_in_control_anchor->GetOwnerNode()->GetName().c_str());
      if (GraphUtils::AddEdge(transdata_peer_out_control_anchor, transdata_peer_in_control_anchor) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                          transdata_peer_out_control_anchor->GetOwnerNode()->GetName().c_str(),
                          transdata_peer_out_control_anchor->GetOwnerNode()->GetType().c_str(),
                          transdata_peer_in_control_anchor->GetOwnerNode()->GetName().c_str(),
                          transdata_peer_in_control_anchor->GetOwnerNode()->GetType().c_str());
        GELOGE(GRAPH_FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
               transdata_peer_out_control_anchor->GetOwnerNode()->GetName().c_str(),
               transdata_peer_out_control_anchor->GetOwnerNode()->GetType().c_str(),
               transdata_peer_in_control_anchor->GetOwnerNode()->GetName().c_str(),
               transdata_peer_in_control_anchor->GetOwnerNode()->GetType().c_str());
        return GRAPH_FAILED;
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::ReLinkOutControlPeerInDataAnchors(
    const NodePtr &transdata_node_keep, const OutDataAnchorPtr &pre_out_anchor,
    const OutControlAnchorPtr &transdata_peer_out_control_anchor) {
  GE_CHECK_NOTNULL(transdata_node_keep);
  GE_CHECK_NOTNULL(pre_out_anchor);
  auto out_control_anchor = transdata_node_keep->GetOutControlAnchor();
  if (out_control_anchor == nullptr) {
    return GRAPH_SUCCESS;
  }
  for (auto &transdata_peer_in_data_anchor : out_control_anchor->GetPeerInDataAnchors()) {
    if (transdata_peer_in_data_anchor == nullptr || transdata_peer_in_data_anchor->GetOwnerNode() == nullptr) {
      continue;
    }
    GELOGD("remove edge.src:%s, dst:%s", transdata_node_keep->GetName().c_str(),
           transdata_peer_in_data_anchor->GetOwnerNode()->GetName().c_str());
    if (GraphUtils::RemoveEdge(out_control_anchor, transdata_peer_in_data_anchor) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Remove control edge between op:%s(%s) and op:%s(%s) failed",
                        out_control_anchor->GetOwnerNode()->GetName().c_str(),
                        out_control_anchor->GetOwnerNode()->GetType().c_str(),
                        transdata_peer_in_data_anchor->GetOwnerNode()->GetName().c_str(),
                        transdata_peer_in_data_anchor->GetOwnerNode()->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Remove][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
             out_control_anchor->GetOwnerNode()->GetName().c_str(),
             out_control_anchor->GetOwnerNode()->GetType().c_str(),
             transdata_peer_in_data_anchor->GetOwnerNode()->GetName().c_str(),
             transdata_peer_in_data_anchor->GetOwnerNode()->GetType().c_str());
      return GRAPH_FAILED;
    }

    if (transdata_peer_out_control_anchor == nullptr) {
      GELOGD("add edge.src:%s, dst:%s", pre_out_anchor->GetOwnerNode()->GetName().c_str(),
             transdata_peer_in_data_anchor->GetOwnerNode()->GetName().c_str());
      if (GraphUtils::AddEdge(pre_out_anchor, transdata_peer_in_data_anchor) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                          pre_out_anchor->GetOwnerNode()->GetName().c_str(),
                          pre_out_anchor->GetOwnerNode()->GetType().c_str(), pre_out_anchor->GetIdx(),
                          transdata_peer_in_data_anchor->GetOwnerNode()->GetName().c_str(),
                          transdata_peer_in_data_anchor->GetOwnerNode()->GetType().c_str(),
                          transdata_peer_in_data_anchor->GetIdx());
        GELOGE(GRAPH_FAILED, "[Add][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
               pre_out_anchor->GetOwnerNode()->GetName().c_str(),
               pre_out_anchor->GetOwnerNode()->GetType().c_str(), pre_out_anchor->GetIdx(),
               transdata_peer_in_data_anchor->GetOwnerNode()->GetName().c_str(),
               transdata_peer_in_data_anchor->GetOwnerNode()->GetType().c_str(),
               transdata_peer_in_data_anchor->GetIdx());
        return GRAPH_FAILED;
      }
    } else {
      GELOGD("add edge.src:%s, dst:%s", transdata_peer_out_control_anchor->GetOwnerNode()->GetName().c_str(),
             transdata_peer_in_data_anchor->GetOwnerNode()->GetName().c_str());
      if (GraphUtils::AddEdge(transdata_peer_out_control_anchor, transdata_peer_in_data_anchor) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                          transdata_peer_out_control_anchor->GetOwnerNode()->GetName().c_str(),
                          transdata_peer_out_control_anchor->GetOwnerNode()->GetType().c_str(),
                          transdata_peer_in_data_anchor->GetOwnerNode()->GetName().c_str(),
                          transdata_peer_in_data_anchor->GetOwnerNode()->GetType().c_str());
        GELOGE(GRAPH_FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
               transdata_peer_out_control_anchor->GetOwnerNode()->GetName().c_str(),
               transdata_peer_out_control_anchor->GetOwnerNode()->GetType().c_str(),
               transdata_peer_in_data_anchor->GetOwnerNode()->GetName().c_str(),
               transdata_peer_in_data_anchor->GetOwnerNode()->GetType().c_str());
        return GRAPH_FAILED;
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::ReLinkTransdataControlOutput2PreNode(
    const NodePtr &transdata_node_keep, const OutDataAnchorPtr &pre_out_anchor,
    const OutControlAnchorPtr &transdata_peer_out_control_anchor) {
  if (ReLinkOutControlPeerInControlAnchors(transdata_node_keep, pre_out_anchor, transdata_peer_out_control_anchor) !=
      GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  return ReLinkOutControlPeerInDataAnchors(transdata_node_keep, pre_out_anchor, transdata_peer_out_control_anchor);
}

graphStatus SameTransdataBreadthFusionPass::Run(ComputeGraphPtr graph) {
  GELOGI("[SameTransdataBreadthFusionPass]: optimize begin.");
  if (graph == nullptr) {
    return GRAPH_SUCCESS;
  }

  for (auto &node : graph->GetDirectNode()) {
    if (IsTransOp(node) || node->GetOutDataNodesSize() <= 1) {
      continue;
    }

    GELOGD("Current normal node name: %s, type: %s.", node->GetName().c_str(), node->GetType().c_str());
    for (auto &out_anchor : node->GetAllOutDataAnchors()) {
      vector<std::vector<pair<OutDataAnchorPtr, InDataAnchorPtr>>> sub_graph_anchors;
      std::vector<pair<OutDataAnchorPtr, InDataAnchorPtr>> nodes_list;
      if (GetSubGraphsBetweenNormalAndTransdataNode(out_anchor, sub_graph_anchors, nodes_list) != GRAPH_SUCCESS) {
        GELOGW("get transop failed!");
        continue;
      }

      if (sub_graph_anchors.size() <= 1) {
        continue;
      }
      sub_graph_anchors_.swap(sub_graph_anchors);

      // check reshape node
      GetSubGraphNodesInfo();
      GELOGD("all trandata node size:%zu", all_transdata_nodes_.size());
      if (ExtractTransNode(graph) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
      }
    }
  }

  GELOGI("[SameTransdataBreadthFusionPass]: Optimize success.");
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::ExtractTransNode(const ComputeGraphPtr &graph) {
  while (all_transdata_nodes_.size() > 1) {
    vector<int> same_transdata_nodes;
    GetSameTransdataNode(same_transdata_nodes);
    GELOGD("same transdata node size:%zu", same_transdata_nodes.size());
    // reuse transdata ,new cast
    if (same_transdata_nodes.size() <= 1) {
      continue;
    }

    int anchors_index = same_transdata_nodes[0];
    auto transdata_in_anchor = sub_graph_anchors_[anchors_index].back().second;
    GE_CHECK_NOTNULL(transdata_in_anchor);
    auto transdata_node_keep = transdata_in_anchor->GetOwnerNode();
    auto transdata_out_anchor = transdata_node_keep->GetOutDataAnchor(0);
    GELOGD("anchor index %d, before transdata node size:%zu", anchors_index,
           before_transdata_nodes_[anchors_index].size());
    if (before_transdata_nodes_[anchors_index].size() > 1) {
      if (RelinkRemainTransdata(graph, same_transdata_nodes) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
      }
    }

    if (LinkNewCastNode2RemainTransdata(graph, same_transdata_nodes, transdata_out_anchor, transdata_node_keep) !=
        GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::RelinkRemainTransdata(const ComputeGraphPtr &graph,
                                                                  const vector<int> &same_transdata_nodes) {
  int anchors_index = same_transdata_nodes[0];
  auto head_node_anchor = sub_graph_anchors_[anchors_index][0].first;
  GE_CHECK_NOTNULL(head_node_anchor);
  auto head_node = head_node_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(head_node->GetOpDesc());
  auto head_output_desc = head_node->GetOpDesc()->GetOutputDescPtr(head_node_anchor->GetIdx());
  auto transdata_in_anchor = sub_graph_anchors_[anchors_index].back().second;
  GE_CHECK_NOTNULL(transdata_in_anchor);
  auto transdata_node_keep = transdata_in_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(transdata_node_keep->GetOpDesc());
  auto transdata_out_anchor = transdata_node_keep->GetOutDataAnchor(0);
  GELOGD("head node:%s, transdata node keep:%s", head_node->GetName().c_str(), transdata_node_keep->GetName().c_str());
  bool reuse_nodes = AllNodeBeforeTransdataHasOneDataOut(anchors_index);
  UpdateTransdataDesc(transdata_in_anchor, transdata_node_keep->GetOpDesc(), head_output_desc);
  auto transdata_peer_out_anchor = sub_graph_anchors_[anchors_index].back().first;
  GE_CHECK_NOTNULL(transdata_peer_out_anchor);
  auto transdata_peer_out_node = transdata_peer_out_anchor->GetOwnerNode();
  GELOGI("remove edge.src:%s, dst:%s", transdata_peer_out_node->GetName().c_str(),
         transdata_node_keep->GetName().c_str());
  if (GraphUtils::RemoveEdge(transdata_peer_out_anchor, transdata_in_anchor) != GRAPH_SUCCESS) {
    GELOGW("remove edge failed!out node %s, in node %s", transdata_peer_out_node->GetName().c_str(),
           transdata_node_keep->GetName().c_str());
  }

  GELOGI("add edge.out node %s, in node %s", head_node->GetName().c_str(), transdata_node_keep->GetName().c_str());
  if (GraphUtils::AddEdge(head_node_anchor, transdata_in_anchor) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                      head_node_anchor->GetOwnerNode()->GetName().c_str(),
                      head_node_anchor->GetOwnerNode()->GetType().c_str(), head_node_anchor->GetIdx(),
                      transdata_in_anchor->GetOwnerNode()->GetName().c_str(),
                      transdata_in_anchor->GetOwnerNode()->GetType().c_str(),
                      transdata_in_anchor->GetIdx());
    GELOGE(GRAPH_FAILED, "[Add][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
           head_node_anchor->GetOwnerNode()->GetName().c_str(), head_node_anchor->GetOwnerNode()->GetType().c_str(),
           head_node_anchor->GetIdx(), transdata_in_anchor->GetOwnerNode()->GetName().c_str(),
           transdata_in_anchor->GetOwnerNode()->GetType().c_str(), transdata_in_anchor->GetIdx());
    return GRAPH_FAILED;
  }

  NodePtr relink_node;
  // relink to transdata output nodes
  if (reuse_nodes) {
    if (ReuseNodesBeforeTransdata(anchors_index, transdata_out_anchor, relink_node) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }

    if (ReLinkTransdataOutput2PreNode(transdata_node_keep, transdata_peer_out_anchor, relink_node) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }
  } else {
    OutDataAnchorPtr pre_out_anchor = transdata_out_anchor;
    if (AddCastNode(graph, same_transdata_nodes[0], pre_out_anchor, relink_node) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }

    if (ReLinkTransdataOutput2PreNode(transdata_node_keep, pre_out_anchor, relink_node) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

void SameTransdataBreadthFusionPass::UpdateTransdataDesc(const InDataAnchorPtr &transdata_in_anchor,
                                                         const OpDescPtr &transdata_op_desc,
                                                         const ConstGeTensorDescPtr &head_output_desc) {
  if (transdata_op_desc == nullptr || transdata_in_anchor == nullptr || head_output_desc == nullptr) {
    return;
  }
  auto mutable_input_desc = transdata_op_desc->MutableInputDesc(transdata_in_anchor->GetIdx());
  GE_CHECK_NOTNULL_JUST_RETURN(mutable_input_desc);
  mutable_input_desc->SetDataType(head_output_desc->GetDataType());
  mutable_input_desc->SetOriginDataType(head_output_desc->GetOriginDataType());
  auto mutable_output_desc = transdata_op_desc->MutableOutputDesc(0);
  GE_CHECK_NOTNULL_JUST_RETURN(mutable_output_desc);
  mutable_output_desc->SetDataType(head_output_desc->GetDataType());
  mutable_output_desc->SetOriginDataType(head_output_desc->GetOriginDataType());
  // maybe need to check support
}

bool SameTransdataBreadthFusionPass::AllNodeBeforeTransdataHasOneDataOut(int anchors_index) {
  for (size_t i = 1; i < before_transdata_nodes_[anchors_index].size(); ++i) {
    auto node = before_transdata_nodes_[anchors_index][i];
    if (node == nullptr) {
      return false;
    }
    if (node->GetOutDataNodes().size() > 1 || node->GetInDataNodes().size() > 1) {
      return false;
    }
  }
  return true;
}

graphStatus SameTransdataBreadthFusionPass::ReuseNodesBeforeTransdata(int anchors_index,
                                                                      const OutDataAnchorPtr &transdata_out_anchor,
                                                                      NodePtr &relink_node) {
  auto head_node_anchor = sub_graph_anchors_[anchors_index][0].first;
  auto head_node_peer_anchor = sub_graph_anchors_[anchors_index][0].second;
  GE_CHECK_NOTNULL(head_node_anchor);
  GE_CHECK_NOTNULL(head_node_peer_anchor);
  GE_CHECK_NOTNULL(transdata_out_anchor);
  GELOGI("remove edge.src:%s, dst:%s", head_node_anchor->GetOwnerNode()->GetName().c_str(),
         head_node_peer_anchor->GetOwnerNode()->GetName().c_str());
  if (head_node_anchor->IsLinkedWith(head_node_peer_anchor)) {
    if (GraphUtils::RemoveEdge(head_node_anchor, head_node_peer_anchor) != GRAPH_SUCCESS) {
      GELOGW("remove edge failed!src:%s, dst:%s", head_node_anchor->GetOwnerNode()->GetName().c_str(),
             head_node_peer_anchor->GetOwnerNode()->GetName().c_str());
    }
  } else {
    GELOGW("edge not link now. src:%s, dst:%s", head_node_anchor->GetOwnerNode()->GetName().c_str(),
           head_node_peer_anchor->GetOwnerNode()->GetName().c_str());
  }

  NodePtr transdata_node_keep = transdata_out_anchor->GetOwnerNode();
  if (before_transdata_nodes_[anchors_index].size() == kNoTransOp) {
    return GRAPH_SUCCESS;
  }
  GELOGI("add edge.src:%s, dst:%s", transdata_node_keep->GetName().c_str(),
         head_node_peer_anchor->GetOwnerNode()->GetName().c_str());
  if (GraphUtils::AddEdge(transdata_out_anchor, head_node_peer_anchor) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                      transdata_out_anchor->GetOwnerNode()->GetName().c_str(),
                      transdata_out_anchor->GetOwnerNode()->GetType().c_str(), transdata_out_anchor->GetIdx(),
                      head_node_peer_anchor->GetOwnerNode()->GetName().c_str(),
                      head_node_peer_anchor->GetOwnerNode()->GetType().c_str(),
                      head_node_peer_anchor->GetIdx());
    GELOGE(GRAPH_FAILED, "[Add][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
           transdata_out_anchor->GetOwnerNode()->GetName().c_str(),
           transdata_out_anchor->GetOwnerNode()->GetType().c_str(), transdata_out_anchor->GetIdx(),
           head_node_peer_anchor->GetOwnerNode()->GetName().c_str(),
           head_node_peer_anchor->GetOwnerNode()->GetType().c_str(), head_node_peer_anchor->GetIdx());
    return GRAPH_FAILED;
  }
  relink_node = head_node_peer_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(transdata_node_keep->GetOpDesc());
  auto transdata_output_desc = transdata_node_keep->GetOpDesc()->GetOutputDescPtr(0);
  GE_CHECK_NOTNULL(transdata_output_desc);
  for (size_t i = 0; i < sub_graph_anchors_[anchors_index].size() - 1; ++i) {
    auto in_data_anchor = sub_graph_anchors_[anchors_index][i].second;
    GE_CHECK_NOTNULL(in_data_anchor);
    auto in_owner_node = in_data_anchor->GetOwnerNode();
    auto in_op_desc = in_owner_node->GetOpDesc();
    GE_CHECK_NOTNULL(in_op_desc);
    auto input_desc = in_op_desc->GetInputDesc(in_data_anchor->GetIdx());
    CopyTensorDesc(transdata_output_desc, input_desc);
    if (in_op_desc->UpdateInputDesc(in_data_anchor->GetIdx(), input_desc) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Update input:%d desc in op:%s(%s) failed", in_data_anchor->GetIdx(),
                        in_op_desc->GetName().c_str(), in_op_desc->GetType().c_str());
      GELOGE(FAILED, "[Update][InputDesc] index:%d in op:%s(%s) failed", in_data_anchor->GetIdx(),
             in_op_desc->GetName().c_str(), in_op_desc->GetType().c_str());
      return FAILED;
    }
    int output_idx = sub_graph_anchors_[anchors_index][i + 1].first->GetIdx();
    auto output_desc = in_op_desc->GetOutputDesc(output_idx);
    CopyTensorDesc(transdata_output_desc, output_desc);
    GE_IF_BOOL_EXEC(in_op_desc->UpdateOutputDesc(output_idx, output_desc) != GRAPH_SUCCESS,
                    REPORT_CALL_ERROR("E19999", "Update output:%d desc in op:%s(%s) failed", output_idx,
                                      in_op_desc->GetName().c_str(), in_op_desc->GetType().c_str());
                    GELOGE(GRAPH_FAILED, "[Update][OutputDesc] index:%d in op:%s(%s) failed", output_idx,
                           in_op_desc->GetName().c_str(), in_op_desc->GetType().c_str());
                    return GRAPH_FAILED);
    // relink control edge
    if (RelinkInControlEdge(in_owner_node, transdata_node_keep) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

void SameTransdataBreadthFusionPass::CopyTensorDesc(const ConstGeTensorDescPtr &src_desc, GeTensorDesc &dst_desc) {
  if (src_desc == nullptr) {
    return;
  }
  dst_desc.SetFormat(src_desc->GetFormat());
  dst_desc.SetOriginFormat(src_desc->GetOriginFormat());
  dst_desc.SetShape(src_desc->GetShape());
  dst_desc.SetOriginShape(src_desc->GetOriginShape());
  uint32_t real_dim = 0;
  if (TensorUtils::GetRealDimCnt(*src_desc, real_dim) == GRAPH_SUCCESS) {
    TensorUtils::SetRealDimCnt(dst_desc, real_dim);
  }
}

graphStatus SameTransdataBreadthFusionPass::LinkNewCastNode2RemainTransdata(
    const ComputeGraphPtr &graph, const vector<int> &same_transdata_nodes, const OutDataAnchorPtr &transdata_out_anchor,
    const NodePtr &transdata_node_keep) {
  for (size_t i = 1; i < same_transdata_nodes.size(); ++i) {
    int anchors_index = same_transdata_nodes[i];
    bool reuse_nodes = AllNodeBeforeTransdataHasOneDataOut(anchors_index);
    auto transdata_peer_out_anchor = sub_graph_anchors_[anchors_index].back().first;
    GE_CHECK_NOTNULL(transdata_peer_out_anchor);
    auto transdata_remove_in_anchor = sub_graph_anchors_[anchors_index].back().second;
    GE_CHECK_NOTNULL(transdata_remove_in_anchor);
    auto transdata_node_remove = transdata_remove_in_anchor->GetOwnerNode();
    if (transdata_node_remove->GetInDataNodes().size() > 1) {
      continue;
    }
    GELOGI("remove edge.src:%s, dst:%s", transdata_peer_out_anchor->GetOwnerNode()->GetName().c_str(),
           transdata_remove_in_anchor->GetOwnerNode()->GetName().c_str());
    if (GraphUtils::RemoveEdge(transdata_peer_out_anchor, transdata_remove_in_anchor) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Remove edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                        transdata_peer_out_anchor->GetOwnerNode()->GetName().c_str(),
                        transdata_peer_out_anchor->GetOwnerNode()->GetType().c_str(),
                        transdata_peer_out_anchor->GetIdx(),
                        transdata_remove_in_anchor->GetOwnerNode()->GetName().c_str(),
                        transdata_remove_in_anchor->GetOwnerNode()->GetType().c_str(),
                        transdata_remove_in_anchor->GetIdx());
      GELOGE(GRAPH_FAILED, "[Remove][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
             transdata_peer_out_anchor->GetOwnerNode()->GetName().c_str(),
             transdata_peer_out_anchor->GetOwnerNode()->GetType().c_str(), transdata_peer_out_anchor->GetIdx(),
             transdata_remove_in_anchor->GetOwnerNode()->GetName().c_str(),
             transdata_remove_in_anchor->GetOwnerNode()->GetType().c_str(), transdata_remove_in_anchor->GetIdx());
      return GRAPH_FAILED;
    }

    OutDataAnchorPtr pre_out_anchor = nullptr;
    NodePtr relink_node = nullptr;
    if (reuse_nodes) {
      // reuse nodes before transdata
      if (ReuseNodesBeforeTransdata(anchors_index, transdata_out_anchor, relink_node) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
      }
      if (before_transdata_nodes_[anchors_index].size() > kNoTransOp) {
        pre_out_anchor = transdata_peer_out_anchor;
      } else {
        pre_out_anchor = transdata_out_anchor;
      }
    } else {
      // miss cast control edge
      pre_out_anchor = transdata_out_anchor;
      if (AddCastNode(graph, same_transdata_nodes[i], pre_out_anchor, relink_node) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
      }
    }

    if (ReLinkTransdataOutput2PreNode(transdata_node_remove, pre_out_anchor, relink_node) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }

    if (RelinkInControlEdge(transdata_node_remove, transdata_node_keep) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }

    if (graph->RemoveNode(transdata_node_remove) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Remove node:%s(%s) from graph:%s failed",
                        transdata_node_remove->GetName().c_str(), transdata_node_remove->GetType().c_str(),
                        graph->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Remove][Node] %s(%s) from graph:%s failed",
             transdata_node_remove->GetName().c_str(), transdata_node_remove->GetType().c_str(),
             graph->GetName().c_str());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::RelinkInControlEdge(const NodePtr &node_src, const NodePtr &node_dst) {
  GE_CHECK_NOTNULL(node_dst);
  GE_CHECK_NOTNULL(node_src);
  if (node_src->GetInControlNodes().empty()) {
    return GRAPH_SUCCESS;
  }
  GE_CHECK_NOTNULL(node_src->GetInControlAnchor());
  for (auto &peer_out_control_anchor : node_src->GetInControlAnchor()->GetPeerOutControlAnchors()) {
    GELOGD("remove edge.src:%s, dst:%s", peer_out_control_anchor->GetOwnerNode()->GetName().c_str(),
           node_src->GetName().c_str());
    if (GraphUtils::RemoveEdge(peer_out_control_anchor, node_src->GetInControlAnchor()) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Remove control edge between op:%s(%s) and op:%s(%s) failed",
                        peer_out_control_anchor->GetOwnerNode()->GetName().c_str(),
                        peer_out_control_anchor->GetOwnerNode()->GetType().c_str(),
                        node_src->GetName().c_str(), node_src->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Remove][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
             peer_out_control_anchor->GetOwnerNode()->GetName().c_str(),
             peer_out_control_anchor->GetOwnerNode()->GetType().c_str(),
             node_src->GetName().c_str(), node_src->GetType().c_str());
      return GRAPH_FAILED;
    }
    GELOGD("add edge.src:%s, dst:%s", peer_out_control_anchor->GetOwnerNode()->GetName().c_str(),
           node_dst->GetName().c_str());
    if (GraphUtils::AddEdge(peer_out_control_anchor, node_dst->GetInControlAnchor()) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                        peer_out_control_anchor->GetOwnerNode()->GetName().c_str(),
                        peer_out_control_anchor->GetOwnerNode()->GetType().c_str(),
                        node_dst->GetName().c_str(), node_dst->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
             peer_out_control_anchor->GetOwnerNode()->GetName().c_str(),
             peer_out_control_anchor->GetOwnerNode()->GetType().c_str(),
             node_dst->GetName().c_str(), node_dst->GetType().c_str());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::AddCastNode(const ComputeGraphPtr &graph, int anchors_index,
                                                        OutDataAnchorPtr &pre_out_anchor, NodePtr &first_link_node) {
  GE_CHECK_NOTNULL(pre_out_anchor);
  GE_CHECK_NOTNULL(graph);
  auto pre_node = pre_out_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(pre_node->GetOpDesc());
  auto pre_output_desc = pre_node->GetOpDesc()->GetOutputDescPtr(pre_out_anchor->GetIdx());
  GE_CHECK_NOTNULL(pre_output_desc);
  for (size_t i = 0; i < sub_graph_anchors_[anchors_index].size() - 1; ++i) {
    auto in_data_anchor = sub_graph_anchors_[anchors_index][i].second;
    GE_CHECK_NOTNULL(in_data_anchor);
    auto in_owner_node = in_data_anchor->GetOwnerNode();
    auto in_op_desc = in_owner_node->GetOpDesc();
    GE_CHECK_NOTNULL(in_op_desc);
    auto input_desc = in_op_desc->GetInputDesc(in_data_anchor->GetIdx());
    input_desc.SetFormat(pre_output_desc->GetFormat());
    input_desc.SetOriginFormat(pre_output_desc->GetOriginFormat());
    input_desc.SetShape(pre_output_desc->GetShape());
    input_desc.SetOriginShape(pre_output_desc->GetOriginShape());
    uint32_t real_dim = 0;
    if (TensorUtils::GetRealDimCnt(*pre_output_desc, real_dim) != GRAPH_SUCCESS) {
      GELOGW("get %s real dim cnt failed!", pre_node->GetName().c_str());
    }
    TensorUtils::SetRealDimCnt(input_desc, real_dim);
    auto output_desc = in_op_desc->GetOutputDesc(sub_graph_anchors_[anchors_index][i + 1].first->GetIdx());
    output_desc.SetFormat(pre_output_desc->GetFormat());
    output_desc.SetOriginFormat(pre_output_desc->GetOriginFormat());
    output_desc.SetShape(pre_output_desc->GetShape());
    output_desc.SetOriginShape(pre_output_desc->GetOriginShape());
    TensorUtils::SetRealDimCnt(output_desc, real_dim);

    auto cast_op_desc = GetCastOp(input_desc, output_desc);
    if (cast_op_desc == nullptr) {
      return GRAPH_FAILED;
    }

    auto cast_node = graph->AddNode(cast_op_desc);
    if (cast_node == nullptr) {
      REPORT_CALL_ERROR("E19999", "Add node:%s(%s) to graph:%s failed",
                        cast_op_desc->GetName().c_str(), cast_op_desc->GetType().c_str(), graph->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Add][Node] %s(%s) to graph:%s failed",
             cast_op_desc->GetName().c_str(), cast_op_desc->GetType().c_str(), graph->GetName().c_str());
      return GRAPH_FAILED;
    }
    GELOGD("add edge.src:%s, dst:%s", pre_out_anchor->GetOwnerNode()->GetName().c_str(), cast_node->GetName().c_str());
    if (GraphUtils::AddEdge(pre_out_anchor, cast_node->GetInDataAnchor(0)) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:0) failed",
                        pre_out_anchor->GetOwnerNode()->GetName().c_str(),
                        pre_out_anchor->GetOwnerNode()->GetType().c_str(), pre_out_anchor->GetIdx(),
                        cast_node->GetName().c_str(), cast_node->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Add][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:0) failed",
             pre_out_anchor->GetOwnerNode()->GetName().c_str(),
             pre_out_anchor->GetOwnerNode()->GetType().c_str(), pre_out_anchor->GetIdx(),
             cast_node->GetName().c_str(), cast_node->GetType().c_str());
      return GRAPH_FAILED;
    }
    if (i == 0) {
      first_link_node = cast_node;
    }

    if (!AttrUtils::SetBool(cast_op_desc, ATTR_NEED_COMPILE, true)) {
      REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NEED_COMPILE.c_str(),
                        cast_op_desc->GetName().c_str(), cast_op_desc->GetType().c_str());
      GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NEED_COMPILE.c_str(),
             cast_op_desc->GetName().c_str(), cast_op_desc->GetType().c_str());
      return FAILED;
    }
    pre_out_anchor = cast_node->GetOutDataAnchor(0);
  }
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::GetSubGraphsBetweenNormalAndTransdataNode(
    OutDataAnchorPtr &out_anchor,
    std::vector<std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>>> &sub_graphs_out,
    std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> &nodes_list) {
  graphStatus ret = GRAPH_SUCCESS;
  if (out_anchor == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param out_anchor is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] out data anchor is null! This should not happen!");
    return GRAPH_FAILED;
  }

  for (auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
    if (peer_in_anchor == nullptr || peer_in_anchor->GetOwnerNode() == nullptr ||
        peer_in_anchor->GetOwnerNode()->GetOpDesc() == nullptr) {
      continue;
    }

    nodes_list.push_back(make_pair(out_anchor, peer_in_anchor));
    auto peer_in_node = peer_in_anchor->GetOwnerNode();
    if ((peer_in_node->GetType() == TRANSDATA && peer_in_node->GetOutDataNodes().size() > 0) ||
        !IsHandleOp(peer_in_node)) {
      sub_graphs_out.push_back(nodes_list);
      nodes_list.pop_back();
    } else {
      if (peer_in_node->GetType() == TRANSDATA) {
        if (peer_in_node->GetOutDataNodes().size() == 0) {
          nodes_list.pop_back();
          continue;
        }
      }
      for (auto &peer_out_anchor : peer_in_node->GetAllOutDataAnchors()) {
        ret = GetSubGraphsBetweenNormalAndTransdataNode(peer_out_anchor, sub_graphs_out, nodes_list);
        if (ret != GRAPH_SUCCESS) {
          GELOGE(GRAPH_FAILED, "[Get][AllTransOp] between normal node failed! node:%s",
                 peer_in_node->GetName().c_str());
          return GRAPH_FAILED;
        }
      }
      nodes_list.pop_back();
    }
  }
  return GRAPH_SUCCESS;
}

bool SameTransdataBreadthFusionPass::IsTransOp(const NodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  return node->GetType() == CAST || node->GetType() == TRANSPOSE || node->GetType() == TRANSPOSED ||
         node->GetType() == RESHAPE || node->GetType() == TRANSDATA;
}

bool SameTransdataBreadthFusionPass::IsHandleOp(const NodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  return node->GetType() == CAST || node->GetType() == TRANSDATA;
}
}  // namespace ge
