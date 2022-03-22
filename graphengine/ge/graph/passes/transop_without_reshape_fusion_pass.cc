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
#include "graph/passes/transop_without_reshape_fusion_pass.h"
#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <atomic>
#include "common/ge/ge_util.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "common/transop_util.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "init/gelib.h"

namespace {
const char *const kRemainNode = "node_remain";
const int kInvalidFusionOpCount = -1;
const char *const kAttrNameSrcFormat = "src_format";
const char *const kAttrNameDstFormat = "dst_format";
}  // namespace

namespace ge {
void TransOpWithoutReshapeFusionPass::SetRemainNode(
  const vector<pair<OutDataAnchorPtr, InDataAnchorPtr>> &nodes_anchor) {
  auto iter = nodes_anchor.begin();
  while (iter != nodes_anchor.end()) {
    auto in_anchor = iter->second;
    if (in_anchor == nullptr) {
      return;
    }
    auto in_node = in_anchor->GetOwnerNode();
    ++iter;
    if (in_node == nullptr) {
      return;
    }
    if (!IsTransOp(in_node)) {
      continue;
    }

    auto op_desc = in_node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    GELOGI("SetRemainNode node is %s", op_desc->GetName().c_str());
    GE_IF_BOOL_EXEC(!op_desc->SetExtAttr(kRemainNode, true),
                    REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", kRemainNode,
                                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
                    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", kRemainNode,
                           op_desc->GetName().c_str(), op_desc->GetType().c_str());
                    return);
  }
}

bool TransOpWithoutReshapeFusionPass::FormatContinuousCheck(const OutDataAnchorPtr &out_anchor,
                                                            const InDataAnchorPtr &in_anchor) {
  if (out_anchor == nullptr || in_anchor == nullptr || in_anchor->GetOwnerNode() == nullptr ||
      out_anchor->GetOwnerNode() == nullptr) {
    return false;
  }
  auto in_node = in_anchor->GetOwnerNode();
  GE_IF_BOOL_EXEC(in_node == nullptr,
                  REPORT_INNER_ERROR("E19999", "Param in_anchor's owner node is nullptr, check invalid");
                  GELOGE(INTERNAL_ERROR, "[Check][Param]Param in_anchor's owner node is nullptr");
                  return false);
  auto in_op = in_node->GetOpDesc();
  auto out_owner_node = out_anchor->GetOwnerNode();
  GE_IF_BOOL_EXEC(out_owner_node == nullptr,
                  REPORT_INNER_ERROR("E19999", "Param out_anchor's owner node is nullptr, check invalid");
                  GELOGE(INTERNAL_ERROR, "[Check][Param] Param out_anchor's owner node is nullptr");
                  return false);
  auto out_op = out_owner_node->GetOpDesc();
  GE_IF_BOOL_EXEC(in_op == nullptr,
                  REPORT_INNER_ERROR("E19999", "Param in_anchor's owner op_desc is nullptr, check invalid");
                  GELOGE(INTERNAL_ERROR, "[Check][Param] Param in_anchor's owner op_desc is nullptr");
                  return false);
  GE_IF_BOOL_EXEC(out_op == nullptr,
                  REPORT_INNER_ERROR("E19999", "Param out_anchor's owner op_desc is nullptr, check invalid");
                  GELOGE(INTERNAL_ERROR, "[Check][Param] Param out_anchor's owner op_desc is nullptr");
                  return false);
  auto in_op_desc = in_op->GetInputDescPtr(in_anchor->GetIdx());
  auto out_op_desc = out_op->GetOutputDescPtr(out_anchor->GetIdx());
  GE_IF_BOOL_EXEC(in_op_desc == nullptr,
                  REPORT_INNER_ERROR("E19999", "Param in_anchor corresponding tensor is nullptr, check invalid");
                  GELOGE(INTERNAL_ERROR, "[Check][Param] Param in_anchor corresponding tensor is nullptr");
                  return false);
  GE_IF_BOOL_EXEC(out_op_desc == nullptr,
                  REPORT_INNER_ERROR("E19999", "Param out_anchor corresponding tensor is nullptr, check invalid");
                  GELOGE(INTERNAL_ERROR, "[Check][Param] Param out_anchor corresponding tensor is nullptr");
                  return false);
  if (!ShapeEqualCheck(in_op_desc->GetShape(), out_op_desc->GetShape())) {
    return false;
  }

  if (in_op->GetType() == CAST || out_op->GetType() == CAST) {
    return TransOpUtil::CheckPrecisionLoss(in_node);
  }

  if (in_op_desc->GetFormat() == FORMAT_ND) {
    return false;
  }

  if (out_op_desc->GetFormat() == FORMAT_ND) {
    return false;
  }

  if (in_op_desc->GetFormat() != out_op_desc->GetFormat()) {
    return false;
  }

  return FusionFormatSupport(in_op_desc->GetFormat());
}

graphStatus TransOpWithoutReshapeFusionPass::GetSubGraphNodesInfo() {
  vector<bool> sub_graph_has_reshape_node(sub_graph_anchors_.size(), false);
  vector<int> transop_num_count(sub_graph_anchors_.size(), 0);
  vector<vector<NodePtr>> sub_graph_nodes(sub_graph_anchors_.size());
  for (size_t i = 0; i < sub_graph_anchors_.size(); ++i) {
    auto nodes_anchor = sub_graph_anchors_[i];
    vector<NodePtr> nodes_tmp;
    auto iter = nodes_anchor.begin();
    auto first_out_anchor = iter->first;
    if (first_out_anchor == nullptr) {
      continue;
    }
    nodes_tmp.push_back(first_out_anchor->GetOwnerNode());
    while (iter != nodes_anchor.end()) {
      auto in_anchor = iter->second;
      GE_CHECK_NOTNULL(in_anchor);
      auto in_node = in_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(in_node);
      if (in_node->GetType() == RESHAPE) {
        sub_graph_has_reshape_node[i] = true;
        break;
      }
      if (in_node->GetType() == TRANSPOSE || in_node->GetType() == TRANSPOSED) {
        auto input_format = in_node->GetOpDesc()->GetInputDescPtr(0)->GetFormat();
        auto output_format = in_node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat();
        if (input_format == output_format) {
          sub_graph_has_reshape_node[i] = true;
          break;
        }
      }

      auto out_anchor = iter->first;
      GE_CHECK_NOTNULL(out_anchor);
      if (!FormatContinuousCheck(out_anchor, in_anchor)) {
        sub_graph_has_reshape_node[i] = true;
        break;
      }

      nodes_tmp.push_back(in_node);
      if (IsTransOp(in_node)) {
        // count transop num
        transop_num_count[i]++;
      }
      ++iter;
    }
    sub_graph_nodes[i].swap(nodes_tmp);
    if (sub_graph_has_reshape_node[i]) {
      SetRemainNode(nodes_anchor);
    }
  }

  sub_graph_has_reshape_node_.swap(sub_graph_has_reshape_node);
  transop_num_count_.swap(transop_num_count);
  sub_graph_nodes_.swap(sub_graph_nodes);
  return GRAPH_SUCCESS;
}

void TransOpWithoutReshapeFusionPass::GetOutDataPeerInControlAnchors(
  const size_t index, vector<vector<InControlAnchorPtr>> &out_data_peer_in_control_anchors) {
  // The caller guarantees that the index is legal.
  for (size_t j = 1; j < sub_graph_anchors_[index].size(); ++j) {
    auto nodes_anchor = sub_graph_anchors_[index][j];
    auto out_data_anchor = nodes_anchor.first;
    GE_CHECK_NOTNULL_JUST_RETURN(out_data_anchor);
    for (const auto &peer_in_control_anchor : out_data_anchor->GetPeerInControlAnchors()) {
      GE_CHECK_NOTNULL_JUST_RETURN(peer_in_control_anchor);
      auto peer_node = peer_in_control_anchor->GetOwnerNode();
      if (peer_node == nullptr) {
        continue;
      }
      auto iter = std::find(sub_graph_nodes_[index].begin(), sub_graph_nodes_[index].end(), peer_node);
      if (iter == sub_graph_nodes_[index].end()) {
        out_data_peer_in_control_anchors[index].push_back(peer_in_control_anchor);
      } else {
        sub_graph_has_out_data_peer_in_control_edge_[index] = true;
      }
    }
  }
}

void TransOpWithoutReshapeFusionPass::GetInControlPeerOutControlAnchors(
  const size_t index, vector<vector<OutControlAnchorPtr>> &in_control_peer_out_control_anchors) {
  // The caller guarantees that the index is legal.
  for (size_t j = 1; j < (sub_graph_nodes_[index].size() - 1); ++j) {
    auto node = sub_graph_nodes_[index][j];
    GE_CHECK_NOTNULL_JUST_RETURN(node);
    auto in_control_anchor = node->GetInControlAnchor();
    if (in_control_anchor == nullptr) {
      continue;
    }

    for (const auto &peer_out_anchor : in_control_anchor->GetPeerOutControlAnchors()) {
      GE_CHECK_NOTNULL_JUST_RETURN(peer_out_anchor);
      auto peer_node = peer_out_anchor->GetOwnerNode();
      if (peer_node == nullptr) {
        continue;
      }
      auto iter = std::find(sub_graph_nodes_[index].begin(), sub_graph_nodes_[index].end(), peer_node);
      if (iter == sub_graph_nodes_[index].end()) {
        in_control_peer_out_control_anchors[index].push_back(peer_out_anchor);
      } else {
        sub_graph_has_control_edge_[index] = true;
      }
    }
  }
}

void TransOpWithoutReshapeFusionPass::GetOutControlPeerAnchors(
  const size_t index, vector<vector<InControlAnchorPtr>> &out_control_peer_in_control_anchors,
  vector<vector<InDataAnchorPtr>> &out_control_peer_in_data_anchors) {
  for (size_t j = 0; j < sub_graph_nodes_[index].size() - 1; ++j) {
    auto node = sub_graph_nodes_[index][j];
    GE_CHECK_NOTNULL_JUST_RETURN(node);
    auto out_control_anchor = node->GetOutControlAnchor();
    GE_CHECK_NOTNULL_JUST_RETURN(out_control_anchor);

    for (const auto &peer_in_anchor : out_control_anchor->GetPeerInControlAnchors()) {
      GE_CHECK_NOTNULL_JUST_RETURN(peer_in_anchor);
      auto peer_node = peer_in_anchor->GetOwnerNode();
      if (peer_node == nullptr) {
        continue;
      }
      auto iter = std::find(sub_graph_nodes_[index].begin(), sub_graph_nodes_[index].end(), peer_node);
      if (iter == sub_graph_nodes_[index].end()) {
        if (j > 0) {
          out_control_peer_in_control_anchors[index].push_back(peer_in_anchor);
        }
      } else {
        sub_graph_has_control_edge_[index] = true;
      }
    }

    for (const auto &peer_in_anchor : out_control_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL_JUST_RETURN(peer_in_anchor);
      auto peer_node = peer_in_anchor->GetOwnerNode();
      if (peer_node == nullptr) {
        continue;
      }
      auto iter = std::find(sub_graph_nodes_[index].begin(), sub_graph_nodes_[index].end(), peer_node);
      if (iter == sub_graph_nodes_[index].end()) {
        if (j > 0) {
          out_control_peer_in_data_anchors[index].push_back(peer_in_anchor);
        }
      } else {
        sub_graph_has_control_edge_[index] = true;
      }
    }
  }
}

void TransOpWithoutReshapeFusionPass::GetControlAnchors() {
  vector<vector<OutControlAnchorPtr>> in_control_peer_out_control_anchors(sub_graph_nodes_.size());
  vector<vector<InControlAnchorPtr>> out_control_peer_in_control_anchors(sub_graph_nodes_.size());
  vector<vector<InDataAnchorPtr>> out_control_peer_in_data_anchors(sub_graph_nodes_.size());
  vector<vector<InControlAnchorPtr>> out_data_peer_in_control_anchors(sub_graph_nodes_.size());
  vector<bool> sub_graph_has_control_edge(sub_graph_nodes_.size(), false);
  sub_graph_has_control_edge_.swap(sub_graph_has_control_edge);
  vector<bool> sub_graph_has_out_data_peer_in_control_edge(sub_graph_nodes_.size(), false);
  sub_graph_has_out_data_peer_in_control_edge_.swap(sub_graph_has_out_data_peer_in_control_edge);
  for (size_t i = 0; i < sub_graph_nodes_.size(); ++i) {
    if (sub_graph_has_reshape_node_[i]) {
      continue;
    }

    GetOutDataPeerInControlAnchors(i, out_data_peer_in_control_anchors);

    GetInControlPeerOutControlAnchors(i, in_control_peer_out_control_anchors);

    GetOutControlPeerAnchors(i, out_control_peer_in_control_anchors, out_control_peer_in_data_anchors);
  }

  in_control_peer_out_control_anchors_.swap(in_control_peer_out_control_anchors);
  out_control_peer_in_control_anchors_.swap(out_control_peer_in_control_anchors);
  out_control_peer_in_data_anchors_.swap(out_control_peer_in_data_anchors);
  out_data_peer_in_control_anchors_.swap(out_data_peer_in_control_anchors);
}

void TransOpWithoutReshapeFusionPass::EraseInvalidAnchorsPair() {
  auto sub_graph_iter = sub_graph_anchors_.begin();
  while (sub_graph_iter != sub_graph_anchors_.end()) {
    if (sub_graph_iter->size() <= 1) {
      sub_graph_iter = sub_graph_anchors_.erase(sub_graph_iter);
    } else {
      ++sub_graph_iter;
    }
  }
}

void TransOpWithoutReshapeFusionPass::UpdateOutputName(const OutDataAnchorPtr &out_anchor,
                                                       const InDataAnchorPtr &old_peer_in_anchor,
                                                       const NodePtr &in_owner_node) {
  if (out_anchor == nullptr || old_peer_in_anchor == nullptr || in_owner_node == nullptr) {
    GELOGI("out_anchor or old_peer_in_anchor or in_owner_node is nullptr");
    return;
  }
  auto out_owner_node = out_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL_JUST_RETURN(out_owner_node);
  GE_CHECK_NOTNULL_JUST_RETURN(old_peer_in_anchor->GetOwnerNode());
  auto old_peer_in_name = old_peer_in_anchor->GetOwnerNode()->GetName();
  auto output_op = out_owner_node->GetOpDesc();
  GE_CHECK_NOTNULL_JUST_RETURN(output_op);
  auto output_names = output_op->GetAllOutputName();
  auto old_peer_in_name_iter = output_names.find(old_peer_in_name);
  if (old_peer_in_name_iter != output_names.end()) {
    output_names.erase(old_peer_in_name_iter);
  }
  output_names[in_owner_node->GetName()] = out_anchor->GetIdx();
  if (!output_op->UpdateOutputName(output_names)) {
    GELOGW("output_op UpdateOutputName failed");
  }
}

void TransOpWithoutReshapeFusionPass::UpdateInputName(const OutDataAnchorPtr &old_peer_out_anchor,
                                                      const InDataAnchorPtr &in_anchor, const NodePtr &out_owner_node) {
  if (old_peer_out_anchor == nullptr || in_anchor == nullptr || out_owner_node == nullptr) {
    GELOGI("old_peer_out_anchor or in_anchor or out_owner_node is nullptr");
    return;
  }
  auto old_node = old_peer_out_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL_JUST_RETURN(old_node);
  auto old_peer_out_name = old_node->GetName();
  auto in_owner_node = in_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL_JUST_RETURN(in_owner_node);
  auto input_op = in_owner_node->GetOpDesc();
  GE_CHECK_NOTNULL_JUST_RETURN(input_op);
  auto input_names = input_op->GetAllInputName();
  auto old_peer_out_name_iter = input_names.find(old_peer_out_name);
  if (old_peer_out_name_iter != input_names.end()) {
    input_names.erase(old_peer_out_name_iter);
  }
  input_names[out_owner_node->GetName()] = in_anchor->GetIdx();
  input_op->UpdateInputName(input_names);
}

graphStatus TransOpWithoutReshapeFusionPass::RelinkSubGraphControlEdges(
  const pair<OutDataAnchorPtr, InDataAnchorPtr> &begin_anchors_pair,
  const pair<OutDataAnchorPtr, InDataAnchorPtr> &end_anchors_pair, const int index) {
  auto out_anchor = begin_anchors_pair.first;
  GE_CHECK_NOTNULL(out_anchor);
  auto out_owner_node = out_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(out_owner_node);
  auto in_anchor = end_anchors_pair.second;
  GE_CHECK_NOTNULL(in_anchor);
  auto in_owner_node = in_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(in_owner_node);
  if (sub_graph_has_control_edge_[index]) {
    GELOGI("add control edge.src:%s, dst:%s", out_owner_node->GetName().c_str(), in_owner_node->GetName().c_str());
    if (GraphUtils::AddEdge(out_owner_node->GetOutControlAnchor(), in_owner_node->GetInControlAnchor()) !=
        GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                        out_owner_node->GetName().c_str(), out_owner_node->GetType().c_str(),
                        in_owner_node->GetName().c_str(), in_owner_node->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
             out_owner_node->GetName().c_str(), out_owner_node->GetType().c_str(),
             in_owner_node->GetName().c_str(), in_owner_node->GetType().c_str());
      return GRAPH_FAILED;
    }
  }

  if (sub_graph_has_out_data_peer_in_control_edge_[index]) {
    GELOGI("add out data 2 in contorl edge.src:%s, dst:%s", out_owner_node->GetName().c_str(),
           in_owner_node->GetName().c_str());
    if (GraphUtils::AddEdge(out_anchor, in_owner_node->GetInControlAnchor()) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                        out_owner_node->GetName().c_str(), out_owner_node->GetType().c_str(),
                        in_owner_node->GetName().c_str(), in_owner_node->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
             out_owner_node->GetName().c_str(), out_owner_node->GetType().c_str(),
             in_owner_node->GetName().c_str(), in_owner_node->GetType().c_str());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus TransOpWithoutReshapeFusionPass::RelinkControlEdgesWhenDescNotChanged(
  const pair<OutDataAnchorPtr, InDataAnchorPtr> &begin_anchors_pair,
  const pair<OutDataAnchorPtr, InDataAnchorPtr> &end_anchors_pair, const int index) {
  if (RelinkSubGraphControlEdges(begin_anchors_pair, end_anchors_pair, index) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  auto out_anchor = begin_anchors_pair.first;
  GE_CHECK_NOTNULL(out_anchor);
  auto out_owner_node = out_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(out_owner_node);
  auto in_anchor = end_anchors_pair.second;
  GE_CHECK_NOTNULL(in_anchor);
  auto in_owner_node = in_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(in_owner_node);
  // can not remove old control edge
  for (const auto &peer_in_anchor : out_control_peer_in_control_anchors_[index]) {
    GE_CHECK_NOTNULL(peer_in_anchor);
    GELOGI("add control edge.src:%s, dst:%s, dst idx:%d", out_owner_node->GetName().c_str(),
           peer_in_anchor->GetOwnerNode()->GetName().c_str(), peer_in_anchor->GetIdx());
    if (GraphUtils::AddEdge(out_owner_node->GetOutControlAnchor(), peer_in_anchor) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                        out_owner_node->GetName().c_str(), out_owner_node->GetType().c_str(),
                        peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                        peer_in_anchor->GetOwnerNode()->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Add]ControlEdge] between op:%s(%s) and op:%s(%s) failed",
             out_owner_node->GetName().c_str(), out_owner_node->GetType().c_str(),
             peer_in_anchor->GetOwnerNode()->GetName().c_str(),
             peer_in_anchor->GetOwnerNode()->GetType().c_str());
      return GRAPH_FAILED;
    }
  }

  for (const auto &peer_out_anchor : in_control_peer_out_control_anchors_[index]) {
    GE_CHECK_NOTNULL(peer_out_anchor);
    GELOGI("add control edge.src:%s, src idx:%d, dst:%s", peer_out_anchor->GetOwnerNode()->GetName().c_str(),
           peer_out_anchor->GetIdx(), in_owner_node->GetName().c_str());
    if (GraphUtils::AddEdge(peer_out_anchor, in_owner_node->GetInControlAnchor()) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                        peer_out_anchor->GetOwnerNode()->GetName().c_str(),
                        peer_out_anchor->GetOwnerNode()->GetType().c_str(),
                        in_owner_node->GetName().c_str(), in_owner_node->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Add]ControlEdge] between op:%s(%s) and op:%s(%s) failed",
             peer_out_anchor->GetOwnerNode()->GetName().c_str(),
             peer_out_anchor->GetOwnerNode()->GetType().c_str(),
             in_owner_node->GetName().c_str(), in_owner_node->GetType().c_str());
      return GRAPH_FAILED;
    }
  }

  for (const auto &peer_in_anchor : out_control_peer_in_data_anchors_[index]) {
    GE_CHECK_NOTNULL(peer_in_anchor);
    GELOGI("add out control 2 in data edge.src:%s, dst:%s, dst idx:%d", out_owner_node->GetName().c_str(),
           peer_in_anchor->GetOwnerNode()->GetName().c_str(), peer_in_anchor->GetIdx());
    if (GraphUtils::AddEdge(out_owner_node->GetOutControlAnchor(), peer_in_anchor) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                        out_owner_node->GetName().c_str(), out_owner_node->GetType().c_str(),
                        peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                        peer_in_anchor->GetOwnerNode()->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Add]ControlEdge] between op:%s(%s) and op:%s(%s) failed",
             out_owner_node->GetName().c_str(), out_owner_node->GetType().c_str(),
             peer_in_anchor->GetOwnerNode()->GetName().c_str(),
             peer_in_anchor->GetOwnerNode()->GetType().c_str());
      return GRAPH_FAILED;
    }
  }

  for (const auto &peer_in_anchor : out_data_peer_in_control_anchors_[index]) {
    GE_CHECK_NOTNULL(peer_in_anchor);
    GELOGI("add out data 2 in control edge.src:%s, dst:%s, dst idx:%d", out_owner_node->GetName().c_str(),
           peer_in_anchor->GetOwnerNode()->GetName().c_str(), peer_in_anchor->GetIdx());
    if (GraphUtils::AddEdge(out_anchor, peer_in_anchor) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                        out_owner_node->GetName().c_str(), out_owner_node->GetType().c_str(),
                        peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                        peer_in_anchor->GetOwnerNode()->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Add]ControlEdge] between op:%s(%s) and op:%s(%s) failed",
             out_owner_node->GetName().c_str(), out_owner_node->GetType().c_str(),
             peer_in_anchor->GetOwnerNode()->GetName().c_str(),
             peer_in_anchor->GetOwnerNode()->GetType().c_str());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus TransOpWithoutReshapeFusionPass::RelinkNodesWhenDescNotChanged(
  const pair<OutDataAnchorPtr, InDataAnchorPtr> &begin_anchors_pair,
  const pair<OutDataAnchorPtr, InDataAnchorPtr> &end_anchors_pair, const int index) {
  auto out_anchor = begin_anchors_pair.first;
  GE_CHECK_NOTNULL(out_anchor);
  auto out_owner_node = out_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(out_owner_node);
  auto in_anchor = end_anchors_pair.second;
  GE_CHECK_NOTNULL(in_anchor);
  auto in_owner_node = in_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(in_owner_node);
  GELOGI("remove edge.src %s, src idx:%d, dst:%s, dst idx:%d",
         end_anchors_pair.first->GetOwnerNode()->GetName().c_str(), end_anchors_pair.first->GetIdx(),
         in_owner_node->GetName().c_str(), in_anchor->GetIdx());
  GE_CHK_STATUS_RET(GraphUtils::RemoveEdge(end_anchors_pair.first, in_anchor),
                    "[Remove][Edge] between %s(%s)(index:%d) and %s(%s)(index:%d) failed",
                    out_owner_node->GetName().c_str(), out_owner_node->GetType().c_str(), out_anchor->GetIdx(),
                    in_owner_node->GetName().c_str(), in_owner_node->GetType().c_str(), in_anchor->GetIdx());
  GELOGI("relink node.src node:%s, src idx:%d, dst node:%s, dst idx:%d", out_owner_node->GetName().c_str(),
         out_anchor->GetIdx(), in_owner_node->GetName().c_str(), in_anchor->GetIdx());
  if (GraphUtils::AddEdge(out_anchor, in_anchor) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                      out_owner_node->GetName().c_str(), out_owner_node->GetType().c_str(), out_anchor->GetIdx(),
                      in_owner_node->GetName().c_str(), in_owner_node->GetType().c_str(), in_anchor->GetIdx());
    GELOGE(GRAPH_FAILED, "[Add][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
           out_owner_node->GetName().c_str(), out_owner_node->GetType().c_str(), out_anchor->GetIdx(),
           in_owner_node->GetName().c_str(), in_owner_node->GetType().c_str(), in_anchor->GetIdx());
    return GRAPH_FAILED;
  } else {
    auto old_peer_in_anchor = begin_anchors_pair.second;
    UpdateOutputName(out_anchor, old_peer_in_anchor, in_owner_node);

    auto old_peer_out_anchor = end_anchors_pair.first;
    UpdateInputName(old_peer_out_anchor, in_anchor, out_owner_node);
  }

  return RelinkControlEdgesWhenDescNotChanged(begin_anchors_pair, end_anchors_pair, index);
}

OpDescPtr TransOpWithoutReshapeFusionPass::GetFormatTransferOp(const GeTensorDesc &format_trans_input_desc,
                                                               const GeTensorDesc &format_trans_output_desc) {
  static std::atomic_long atomic_fusion_format_transfer_op_count(1);
  auto fusion_format_transfer_op_count = atomic_fusion_format_transfer_op_count.fetch_add(1);

  std::stringstream format_transfer_op_name;
  format_transfer_op_name << "fusion_format_transfer_" << fusion_format_transfer_op_count;
  OpDescPtr format_transfer_op = MakeShared<OpDesc>(format_transfer_op_name.str().c_str(), TRANSDATA);
  if (format_transfer_op == nullptr) {
    REPORT_CALL_ERROR("E19999", "New OpDesc failed");
    GELOGE(INTERNAL_ERROR, "[New][OpDesc] failed");
    return nullptr;
  }

  GE_IF_BOOL_EXEC(!AttrUtils::SetInt(format_transfer_op, ATTR_NAME_INPUT_FORMAT,
                                     static_cast<int64_t>(format_trans_input_desc.GetFormat())),
                  REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_INPUT_FORMAT.c_str(),
                                    format_transfer_op->GetName().c_str(), format_transfer_op->GetType().c_str());
                  GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_INPUT_FORMAT.c_str(),
                         format_transfer_op->GetName().c_str(), format_transfer_op->GetType().c_str());
                  return nullptr);
  GE_IF_BOOL_EXEC(!AttrUtils::SetInt(format_transfer_op, ATTR_NAME_OUTPUT_FORMAT,
                                     static_cast<int64_t>(format_trans_output_desc.GetFormat())),
                  REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_OUTPUT_FORMAT.c_str(),
                                    format_transfer_op->GetName().c_str(), format_transfer_op->GetType().c_str());
                  GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_OUTPUT_FORMAT.c_str(),
                         format_transfer_op->GetName().c_str(), format_transfer_op->GetType().c_str());
                  return nullptr);

  string src_format = TypeUtils::FormatToSerialString(format_trans_input_desc.GetFormat());
  string dst_format = TypeUtils::FormatToSerialString(format_trans_output_desc.GetFormat());

  GE_IF_BOOL_EXEC(!AttrUtils::SetStr(format_transfer_op, kAttrNameSrcFormat, src_format),
                  REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", kAttrNameSrcFormat,
                                    format_transfer_op->GetName().c_str(), format_transfer_op->GetType().c_str());
                  GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", kAttrNameSrcFormat,
                         format_transfer_op->GetName().c_str(), format_transfer_op->GetType().c_str());
                  return nullptr);

  GE_IF_BOOL_EXEC(!AttrUtils::SetStr(format_transfer_op, kAttrNameDstFormat, dst_format),
                  REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", kAttrNameDstFormat,
                                    format_transfer_op->GetName().c_str(), format_transfer_op->GetType().c_str());
                  GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", kAttrNameDstFormat,
                         format_transfer_op->GetName().c_str(), format_transfer_op->GetType().c_str());
                  return nullptr);

  GE_IF_BOOL_EXEC(format_transfer_op->AddInputDesc(format_trans_input_desc) != GRAPH_SUCCESS,
                  REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                                    format_transfer_op->GetName().c_str(), format_transfer_op->GetType().c_str());
                  GELOGE(INTERNAL_ERROR, "[Add][InputDesc] to op:%s(%s) failed",
                         format_transfer_op->GetName().c_str(), format_transfer_op->GetType().c_str());
                  return nullptr);

  GE_IF_BOOL_EXEC(format_transfer_op->AddOutputDesc(format_trans_output_desc) != GRAPH_SUCCESS,
                  REPORT_CALL_ERROR("E19999", "Add ouput desc to op:%s(%s) failed",
                                    format_transfer_op->GetName().c_str(), format_transfer_op->GetType().c_str());
                  GELOGE(INTERNAL_ERROR, "[Add][OutputDesc] to op:%s(%s) failed",
                         format_transfer_op->GetName().c_str(), format_transfer_op->GetType().c_str());
                  return nullptr);

  GE_IF_BOOL_EXEC(!ge::AttrUtils::SetBool(format_transfer_op, ATTR_NEED_COMPILE, true),
                  REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NEED_COMPILE.c_str(),
                                    format_transfer_op->GetName().c_str(), format_transfer_op->GetType().c_str());
                  GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NEED_COMPILE.c_str(),
                         format_transfer_op->GetName().c_str(), format_transfer_op->GetType().c_str());
                  return nullptr);
  return format_transfer_op;
}

OpDescPtr TransOpWithoutReshapeFusionPass::GetCastOp(const GeTensorDesc &cast_input_desc,
                                                     const GeTensorDesc &cast_output_desc) {
  static std::atomic_long atomic_fusion_cast_op_count(1);
  auto fusion_cast_op_count = atomic_fusion_cast_op_count.fetch_add(1);

  std::stringstream cast_op_name;
  cast_op_name << "fusion_cast_op_" << fusion_cast_op_count;
  auto node_op = ge::OperatorFactory::CreateOperator(cast_op_name.str(), CAST);
  auto cast_op = ge::OpDescUtils::GetOpDescFromOperator(node_op);
  node_op.BreakConnect();
  if (cast_op == nullptr) {
    REPORT_CALL_ERROR("E19999", "Create operator:%s(%s) failed", cast_op_name.str().c_str(), CAST);
    GELOGE(INTERNAL_ERROR, "[Create][Operator] %s(%s) failed", cast_op_name.str().c_str(), CAST);
    return nullptr;
  }
  const int default_input_index = 0;
  const int default_output_index = 0;
  if (cast_op->GetInputsSize() == 0) {
    GE_IF_BOOL_EXEC(cast_op->AddInputDesc(cast_input_desc) != GRAPH_SUCCESS,
                    REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                                      cast_op->GetName().c_str(), cast_op->GetType().c_str());
                    GELOGE(INTERNAL_ERROR, "[Add][InputDesc] to op:%s(%s) failed",
                           cast_op->GetName().c_str(), cast_op->GetType().c_str());
                    return nullptr);
  } else {
    GE_IF_BOOL_EXEC(cast_op->UpdateInputDesc(default_input_index, cast_input_desc) != GRAPH_SUCCESS,
                    REPORT_CALL_ERROR("E19999", "Update input:%d desc of op:%s(%s) failed", default_input_index,
                                      cast_op->GetName().c_str(), cast_op->GetType().c_str());
                    GELOGE(INTERNAL_ERROR, "[Update][InputDesc] of op:%s(%s) failed, input index:%d",
                           cast_op->GetName().c_str(), cast_op->GetType().c_str(), default_input_index);
                    return nullptr);
  }

  if (cast_op->GetOutputsSize() == 0) {
    GE_IF_BOOL_EXEC(cast_op->AddOutputDesc(cast_output_desc) != GRAPH_SUCCESS,
                    REPORT_CALL_ERROR("E19999", "Add output desc to op:%s(%s) failed",
                                      cast_op->GetName().c_str(), cast_op->GetType().c_str());
                    GELOGE(INTERNAL_ERROR, "[Add][OutputDesc] to op:%s(%s) failed",
                           cast_op->GetName().c_str(), cast_op->GetType().c_str());
                    return nullptr);
  } else {
    GE_IF_BOOL_EXEC(cast_op->UpdateOutputDesc(default_output_index, cast_output_desc) != GRAPH_SUCCESS,
                    REPORT_CALL_ERROR("E19999", "Update output:%d desc of op:%s(%s) failed", default_output_index,
                                      cast_op->GetName().c_str(), cast_op->GetType().c_str());
                    GELOGE(INTERNAL_ERROR, "[Update][OutputDesc] of op:%s(%s) failed, output index:%d",
                           cast_op->GetName().c_str(), cast_op->GetType().c_str(), default_output_index);
                    return nullptr);
  }

  if (!AttrUtils::SetInt(cast_op, CAST_ATTR_DST_TYPE, static_cast<int64_t>(cast_output_desc.GetDataType()))) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", CAST_ATTR_DST_TYPE.c_str(),
                      cast_op->GetName().c_str(), cast_op->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", CAST_ATTR_DST_TYPE.c_str(),
           cast_op->GetName().c_str(), cast_op->GetType().c_str());
    return nullptr;
  }
  if (!AttrUtils::SetBool(cast_op, ATTR_NEED_COMPILE, true)) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NEED_COMPILE.c_str(),
                      cast_op->GetName().c_str(), cast_op->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NEED_COMPILE.c_str(),
           cast_op->GetName().c_str(), cast_op->GetType().c_str());
    return nullptr;
  }
  return cast_op;
}

bool TransOpWithoutReshapeFusionPass::InsertCastFirstCheck(const GeTensorDesc &out_desc,
                                                           const GeTensorDesc &in_desc) const {
  return out_desc.GetDataType() != in_desc.GetDataType() && out_desc.GetDataType() != DT_FLOAT16 &&
         in_desc.GetDataType() == DT_FLOAT16;
}

void TransOpWithoutReshapeFusionPass::GetFormatTransferDesc(const GeTensorDesc &out_desc, const GeTensorDesc &in_desc,
                                                            GeTensorDesc &format_transfer_input,
                                                            GeTensorDesc &format_transfer_output) {
  bool insert_cast_first = InsertCastFirstCheck(out_desc, in_desc);
  if (insert_cast_first) {
    format_transfer_input = out_desc;
    format_transfer_input.SetDataType(in_desc.GetDataType());
    format_transfer_output = in_desc;
  } else {
    format_transfer_input = out_desc;
    format_transfer_output = in_desc;
    format_transfer_output.SetDataType(out_desc.GetDataType());
  }
}

void TransOpWithoutReshapeFusionPass::GetCastOpDesc(const GeTensorDesc &out_desc, const GeTensorDesc &in_desc,
                                                    GeTensorDesc &cast_input, GeTensorDesc &cast_output) {
  bool insert_cast_first = InsertCastFirstCheck(out_desc, in_desc);
  if (insert_cast_first) {
    cast_input = out_desc;
    cast_output = out_desc;
    cast_output.SetDataType(in_desc.GetDataType());
  } else {
    cast_input = in_desc;
    cast_input.SetDataType(out_desc.GetDataType());
    cast_output = in_desc;
  }
}

void TransOpWithoutReshapeFusionPass::GetBeginOutDescAndEndInDesc(const int index, GeTensorDesc &out_desc,
                                                                  GeTensorDesc &in_desc) {
  auto nodes_anchor = sub_graph_anchors_[index];
  auto out_peer_anchor = nodes_anchor.front().second;
  GE_CHECK_NOTNULL_JUST_RETURN(out_peer_anchor);
  auto out_owner_node = out_peer_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL_JUST_RETURN(out_owner_node);
  auto out_peer_op_desc = out_owner_node->GetOpDesc();
  GE_IF_BOOL_EXEC(out_peer_op_desc == nullptr,
                  GELOGE(INTERNAL_ERROR, "[Get][OpDesc] failed, out_peer_op_desc is nullptr"); return);
  out_desc = out_peer_op_desc->GetInputDesc(out_peer_anchor->GetIdx());

  auto in_peer_anchor = nodes_anchor.back().first;
  GE_CHECK_NOTNULL_JUST_RETURN(in_peer_anchor);
  auto in_owner_node = in_peer_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL_JUST_RETURN(in_owner_node);
  auto in_peer_op_desc = in_owner_node->GetOpDesc();
  GE_IF_BOOL_EXEC(in_peer_op_desc == nullptr,
                  GELOGE(INTERNAL_ERROR, "[Get][OpDesc] failed, in_peer_op_desc is nullptr"); return);
  in_desc = in_peer_op_desc->GetOutputDesc(in_peer_anchor->GetIdx());
}

graphStatus TransOpWithoutReshapeFusionPass::FormatFusion(const int index, OpDescPtr &format_transfer_op,
                                                          int32_t &fusion_op_count, bool &fusion_continue) {
  GeTensorDesc out_desc;
  GeTensorDesc in_desc;
  GetBeginOutDescAndEndInDesc(index, out_desc, in_desc);

  GeTensorDesc format_transfer_input;
  GeTensorDesc format_transfer_output;
  GetFormatTransferDesc(out_desc, in_desc, format_transfer_input, format_transfer_output);

  if (out_desc.GetFormat() == in_desc.GetFormat() &&
      (!ShapeEqualCheck(out_desc.GetShape(), in_desc.GetShape()) ||
       !ShapeEqualCheck(out_desc.GetOriginShape(), in_desc.GetOriginShape()))) {
    SetRemainNode(sub_graph_anchors_[index]);
    return GRAPH_SUCCESS;
  }

  if (out_desc.GetFormat() != in_desc.GetFormat() && FusionFormatSupport(out_desc.GetFormat()) &&
      FusionFormatSupport(in_desc.GetFormat())) {
    // create format transop
    format_transfer_op = GetFormatTransferOp(format_transfer_input, format_transfer_output);
    if (format_transfer_op == nullptr) {
      return GRAPH_FAILED;
    }

    if (OpAccuracyAbilityCheck(format_transfer_op)) {
      ++fusion_op_count;
      GELOGI("support format transfer op %s", format_transfer_op->GetName().c_str());
    } else {
      GELOGW("ability not support.src format:%d, src datatype:%d, dst format:%d, dst datatype:%d",
             format_transfer_input.GetFormat(), format_transfer_input.GetDataType(), format_transfer_output.GetFormat(),
             format_transfer_output.GetDataType());
      fusion_op_count = kInvalidFusionOpCount;
    }
  } else if (out_desc.GetFormat() != in_desc.GetFormat()) {
    SetRemainNode(sub_graph_anchors_[index]);
    return GRAPH_SUCCESS;
  }
  fusion_continue = true;
  return GRAPH_SUCCESS;
}

graphStatus TransOpWithoutReshapeFusionPass::DataTypeFusion(const int index, OpDescPtr &cast_op,
                                                            int32_t &fusion_op_count) {
  GeTensorDesc out_desc;
  GeTensorDesc in_desc;
  GetBeginOutDescAndEndInDesc(index, out_desc, in_desc);

  GeTensorDesc cast_input;
  GeTensorDesc cast_output;
  GetCastOpDesc(out_desc, in_desc, cast_input, cast_output);

  if (fusion_op_count != kInvalidFusionOpCount && out_desc.GetDataType() != in_desc.GetDataType()) {
    // create cast op
    cast_op = GetCastOp(cast_input, cast_output);
    if (cast_op == nullptr) {
      fusion_op_count = kInvalidFusionOpCount;
      return GRAPH_FAILED;
    }

    if (OpAccuracyAbilityCheck(cast_op)) {
      ++fusion_op_count;
      GELOGI("support cast op %s. src format:%d, src datatype:%d, dst format:%d, dst datatype:%d",
             cast_op->GetName().c_str(), cast_input.GetFormat(), cast_input.GetDataType(), cast_output.GetFormat(),
             cast_output.GetDataType());
    } else {
      GELOGW("ability not support.src format:%d, src datatype:%d, dst format:%d, dst datatype:%d",
             cast_input.GetFormat(), cast_input.GetDataType(), cast_output.GetFormat(), cast_output.GetDataType());
      fusion_op_count = kInvalidFusionOpCount;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus TransOpWithoutReshapeFusionPass::TransOpFuseHandle(const ComputeGraphPtr &graph, const int index) {
  bool fusion_continue = false;
  OpDescPtr format_transfer_op = nullptr;
  int32_t fusion_op_count = 0;
  auto fortmat_fusion_ret = FormatFusion(index, format_transfer_op, fusion_op_count, fusion_continue);
  if (fortmat_fusion_ret != GRAPH_SUCCESS || !fusion_continue) {
    SetRemainNode(sub_graph_anchors_[index]);
    return GRAPH_SUCCESS;
  }

  OpDescPtr cast_op = nullptr;
  if (DataTypeFusion(index, cast_op, fusion_op_count) != GRAPH_SUCCESS) {
    SetRemainNode(sub_graph_anchors_[index]);
    return GRAPH_SUCCESS;
  }

  if (fusion_op_count != kInvalidFusionOpCount && fusion_op_count < transop_num_count_[index]) {
    GeTensorDesc out_desc;
    GeTensorDesc in_desc;
    GetBeginOutDescAndEndInDesc(index, out_desc, in_desc);
    bool insert_cast_first = InsertCastFirstCheck(out_desc, in_desc);
    if (InsertNewTransOp(graph, cast_op, format_transfer_op, index, insert_cast_first) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }
  } else {
    // remain all nodes
    SetRemainNode(sub_graph_anchors_[index]);
  }
  return GRAPH_SUCCESS;
}

void TransOpWithoutReshapeFusionPass::RemoveNousedNodes(const ComputeGraphPtr &graph) {
  if (graph == nullptr) {
    return;
  }
  for (size_t i = 0; i < sub_graph_nodes_.size(); ++i) {
    if (sub_graph_has_reshape_node_[i]) {
      continue;
    }

    for (const auto &node : sub_graph_nodes_[i]) {
      GE_CHECK_NOTNULL_JUST_RETURN(node);
      // remove nodes
      if (!IsTransOp(node)) {
        continue;
      }

      auto op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL_JUST_RETURN(op_desc);
      bool node_remain_flag = op_desc->TryGetExtAttr(kRemainNode, false);
      if (node_remain_flag) {
        continue;
      }

      GE_IF_BOOL_EXEC(!op_desc->SetExtAttr(kRemainNode, true),
                      GELOGE(INTERNAL_ERROR, "[Set][ExtAttr] for op:%s failed", op_desc->GetName().c_str()); return);
      GELOGI("remove node:%s", node->GetName().c_str());
      if (GraphUtils::IsolateNode(node, {0}) != GRAPH_SUCCESS) {
        GELOGW("Isolate node: %s failed.", node->GetName().c_str());
        continue;
      }
      if (GraphUtils::RemoveNodeWithoutRelink(graph, node) != GRAPH_SUCCESS) {
        GELOGW("Remove node: %s failed.", node->GetName().c_str());
        continue;
      }
    }
  }
}

graphStatus TransOpWithoutReshapeFusionPass::Run(ComputeGraphPtr graph) {
  GELOGI("[TransOpWithoutReshapeFusionPass]: optimize begin.");
  if (graph == nullptr) {
    return GRAPH_SUCCESS;
  }

  for (const auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (IsTransOp(node)) {
      continue;
    }
    bool is_unknown = false;
    auto ret = NodeUtils::GetNodeUnknownShapeStatus(*node, is_unknown);
    if (ret != GRAPH_SUCCESS) {
      GELOGW("Get node unknown status failed, node name:%s, type:%s.", node->GetName().c_str(),
             node->GetType().c_str());
      continue;
    }
    if (is_unknown) {
      GELOGI("Current node %s, type %s is unknown shape which should be skip.", node->GetName().c_str(),
             node->GetType().c_str());
      continue;
    }
    GELOGI("Current normal node name: %s, type: %s.", node->GetName().c_str(), node->GetType().c_str());
    for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
      GE_CHECK_NOTNULL(out_anchor);
      vector<vector<pair<OutDataAnchorPtr, InDataAnchorPtr>>> sub_graph_anchors;
      vector<pair<OutDataAnchorPtr, InDataAnchorPtr>> nodes_list;
      if (GetSubGraphsBetweenNormalNode(out_anchor, sub_graph_anchors, nodes_list) != GRAPH_SUCCESS) {
        GELOGW("get transops failed!");
        continue;
      }

      sub_graph_anchors_.swap(sub_graph_anchors);
      EraseInvalidAnchorsPair();
      if (sub_graph_anchors_.empty()) {
        continue;
      }

      // check reshape node
      if (GetSubGraphNodesInfo() != GRAPH_SUCCESS) {
        continue;
      }

      // save control edge
      GetControlAnchors();

      if (TransOpFuse(graph) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
      }
    }
  }
  GELOGI("[TransOpWithoutReshapeFusionPass]: Optimize end.");
  return GRAPH_SUCCESS;
}

bool TransOpWithoutReshapeFusionPass::DescEqualCheck(ConstGeTensorDescPtr &desc_src,
                                                     ConstGeTensorDescPtr &desc_dst) const {
  if (desc_src == nullptr || desc_dst == nullptr) {
    return false;
  }
  if (desc_src->GetFormat() != desc_dst->GetFormat() || desc_src->GetDataType() != desc_dst->GetDataType()) {
    return false;
  }

  if (!ShapeEqualCheck(desc_src->GetShape(), desc_dst->GetShape())) {
    return false;
  }

  return ShapeEqualCheck(desc_src->GetOriginShape(), desc_dst->GetOriginShape());
}

bool TransOpWithoutReshapeFusionPass::ShapeEqualCheck(const GeShape &src, const GeShape &dst) const {
  if (src.GetDims().size() != dst.GetDims().size()) {
    return false;
  }

  for (size_t i = 0; i < src.GetDims().size(); ++i) {
    if (src.GetDim(i) != dst.GetDim(i)) {
      return false;
    }
  }
  return true;
}

graphStatus TransOpWithoutReshapeFusionPass::TransOpFuse(const ComputeGraphPtr &graph) {
  for (size_t i = 0; i < sub_graph_anchors_.size(); ++i) {
    if (sub_graph_has_reshape_node_[i]) {
      continue;
    }

    auto nodes_anchor = sub_graph_anchors_[i];
    auto out_anchor = nodes_anchor.front().first;
    GE_CHECK_NOTNULL(out_anchor);
    auto out_op_desc = out_anchor->GetOwnerNode()->GetOpDesc();
    GE_CHECK_NOTNULL(out_op_desc);
    auto out_desc = out_op_desc->GetOutputDescPtr(out_anchor->GetIdx());
    GE_CHECK_NOTNULL(out_desc);
    auto in_anchor = nodes_anchor.back().second;
    GE_CHECK_NOTNULL(in_anchor);
    auto in_op_desc = in_anchor->GetOwnerNode()->GetOpDesc();
    GE_CHECK_NOTNULL(in_op_desc);
    auto in_desc = in_op_desc->GetInputDescPtr(in_anchor->GetIdx());
    GE_CHECK_NOTNULL(in_desc);
    if (FusionFormatSupport(out_desc->GetFormat()) && DescEqualCheck(out_desc, in_desc)) {
      // relink begin_out to end_in
      if (RelinkNodesWhenDescNotChanged(nodes_anchor.front(), nodes_anchor.back(), static_cast<int>(i)) !=
          GRAPH_SUCCESS) {
        return GRAPH_FAILED;
      }
    } else {
      if (TransOpFuseHandle(graph, static_cast<int>(i)) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
      }
    }
  }
  RemoveNousedNodes(graph);
  return GRAPH_SUCCESS;
}

graphStatus TransOpWithoutReshapeFusionPass::AddTransNode(const ComputeGraphPtr &graph, const OpDescPtr &transop,
                                                          NodePtr &trans_node) {
  if (graph == nullptr) {
    return GRAPH_SUCCESS;
  }
  if (transop == nullptr) {
    return GRAPH_SUCCESS;
  }

  trans_node = graph->AddNode(transop);
  if (trans_node == nullptr) {
    REPORT_CALL_ERROR("E19999", "Add node:%s(%s) to graph:%s failed",
                      transop->GetName().c_str(), transop->GetType().c_str(), graph->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Add][Node] %s(%s) to graph:%s failed",
           transop->GetName().c_str(), transop->GetType().c_str(), graph->GetName().c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus TransOpWithoutReshapeFusionPass::GetTransNode(const ComputeGraphPtr &graph, const OpDescPtr &cast_op,
                                                          const OpDescPtr &format_transfer_op,
                                                          const bool insert_cast_first,
                                                          std::vector<NodePtr> &new_trans_nodes) {
  NodePtr format_transfer_node;
  if (AddTransNode(graph, format_transfer_op, format_transfer_node) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  NodePtr cast_node;
  if (AddTransNode(graph, cast_op, cast_node) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  if (insert_cast_first) {
    if (cast_node != nullptr) {
      new_trans_nodes.push_back(cast_node);
    }
    if (format_transfer_node != nullptr) {
      new_trans_nodes.push_back(format_transfer_node);
    }
  } else {
    if (format_transfer_node != nullptr) {
      new_trans_nodes.push_back(format_transfer_node);
    }
    if (cast_node != nullptr) {
      new_trans_nodes.push_back(cast_node);
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus TransOpWithoutReshapeFusionPass::InsertNewTransOp(const ComputeGraphPtr &graph, const OpDescPtr &cast_op,
                                                              const OpDescPtr &format_transfer_op, const int index,
                                                              const bool insert_cast_first) {
  std::vector<NodePtr> new_trans_nodes;
  if (GetTransNode(graph, cast_op, format_transfer_op, insert_cast_first, new_trans_nodes) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  if (new_trans_nodes.empty()) {
    GELOGI("No new trans node. Do not need insert new transop.");
    return GRAPH_SUCCESS;
  }

  pair<OutDataAnchorPtr, InDataAnchorPtr> begin_out = sub_graph_anchors_[index].front();
  pair<OutDataAnchorPtr, InDataAnchorPtr> end_in = sub_graph_anchors_[index].back();
  auto out_anchor = begin_out.first;
  GE_CHECK_NOTNULL(out_anchor);
  auto out_owner_node = out_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(out_owner_node);
  auto in_anchor = end_in.second;
  GE_CHECK_NOTNULL(in_anchor);
  auto in_owner_node = in_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(in_owner_node);
  GELOGI("remove edge.src:%s, src idx:%d, dst:%s, dst idx:%d", end_in.first->GetOwnerNode()->GetName().c_str(),
         end_in.first->GetIdx(), in_anchor->GetOwnerNode()->GetName().c_str(), in_anchor->GetIdx());
  GE_CHK_STATUS_RET(GraphUtils::RemoveEdge(end_in.first, in_anchor),
                    "[Remove][Edge] between %s and %s failed",
                    out_owner_node->GetName().c_str(), in_owner_node->GetName().c_str());
  GELOGI("add edge.src:%s, src idx:%d, dst:%s", out_anchor->GetOwnerNode()->GetName().c_str(), out_anchor->GetIdx(),
         new_trans_nodes.front()->GetName().c_str());
  if (GraphUtils::AddEdge(out_anchor, new_trans_nodes.front()->GetInAnchor(0)) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:0) failed",
                      out_owner_node->GetName().c_str(), out_owner_node->GetType().c_str(), out_anchor->GetIdx(),
                      new_trans_nodes.front()->GetName().c_str(), new_trans_nodes.front()->GetType().c_str());
    GELOGE(GRAPH_FAILED, "[Add][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:0) failed",
           out_owner_node->GetName().c_str(), out_owner_node->GetType().c_str(), out_anchor->GetIdx(),
           new_trans_nodes.front()->GetName().c_str(), new_trans_nodes.front()->GetType().c_str());
    return GRAPH_FAILED;
  } else {
    auto old_peer_in_anchor = begin_out.second;
    GE_CHECK_NOTNULL(old_peer_in_anchor);
    UpdateOutputName(out_anchor, old_peer_in_anchor, in_owner_node);
  }

  if (new_trans_nodes.size() > 1) {
    GELOGI("add edge.src:%s, dst:%s", new_trans_nodes.front()->GetName().c_str(),
           new_trans_nodes.back()->GetName().c_str());
    if (GraphUtils::AddEdge(new_trans_nodes.front()->GetOutAnchor(0), new_trans_nodes.back()->GetInAnchor(0)) !=
        GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:0) failed",
                        new_trans_nodes.front()->GetName().c_str(), new_trans_nodes.front()->GetType().c_str(),
                        new_trans_nodes.back()->GetName().c_str(), new_trans_nodes.back()->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Add][Edge] between op:%s(%s)(index:0) and op:%s(%s)(index:0) failed",
             new_trans_nodes.front()->GetName().c_str(), new_trans_nodes.front()->GetType().c_str(),
             new_trans_nodes.back()->GetName().c_str(), new_trans_nodes.back()->GetType().c_str());
      return GRAPH_FAILED;
    } else {
      auto old_peer_out_anchor = end_in.first;
      GE_CHECK_NOTNULL(old_peer_out_anchor);
      UpdateInputName(old_peer_out_anchor, in_anchor, out_owner_node);
    }
  }
  GELOGI("add edge.src:%s, dst:%s, dst idx:%d", new_trans_nodes.back()->GetName().c_str(),
         in_anchor->GetOwnerNode()->GetName().c_str(), in_anchor->GetIdx());
  if (GraphUtils::AddEdge(new_trans_nodes.back()->GetOutAnchor(0), in_anchor) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:%d) failed",
                      new_trans_nodes.front()->GetName().c_str(), new_trans_nodes.front()->GetType().c_str(),
                      in_owner_node->GetName().c_str(), in_owner_node->GetType().c_str(), in_anchor->GetIdx());
    GELOGE(GRAPH_FAILED, "[Add][Edge] between op:%s(%s)(index:0) and op:%s(%s)(index:%d) failed",
           new_trans_nodes.front()->GetName().c_str(), new_trans_nodes.front()->GetType().c_str(),
           in_owner_node->GetName().c_str(), in_owner_node->GetType().c_str(), in_anchor->GetIdx());
    return GRAPH_FAILED;
  }

  return RelinkControlEdge(index, out_anchor, new_trans_nodes);
}

graphStatus TransOpWithoutReshapeFusionPass::RelinkControlEdge(const int index, const OutDataAnchorPtr &out_anchor,
                                                               const vector<NodePtr> &new_trans_nodes) {
  GE_CHECK_NOTNULL(out_anchor);
  if (new_trans_nodes.front() == nullptr || new_trans_nodes.back() == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param new_trans_nodes front or back is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Param new_trans_nodes front or back is nullptr");
    return GRAPH_FAILED;
  }
  if (sub_graph_has_control_edge_[index]) {
    GELOGI("add control edge.src:%s, dst:%s", out_anchor->GetOwnerNode()->GetName().c_str(),
           new_trans_nodes.front()->GetName().c_str());
    if (GraphUtils::AddEdge(out_anchor->GetOwnerNode()->GetOutControlAnchor(),
                            new_trans_nodes.front()->GetInControlAnchor()) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                        out_anchor->GetOwnerNode()->GetName().c_str(), out_anchor->GetOwnerNode()->GetType().c_str(),
                        new_trans_nodes.front()->GetName().c_str(), new_trans_nodes.front()->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
             out_anchor->GetOwnerNode()->GetName().c_str(), out_anchor->GetOwnerNode()->GetType().c_str(),
             new_trans_nodes.front()->GetName().c_str(), new_trans_nodes.front()->GetType().c_str());
      return GRAPH_FAILED;
    }
  }

  for (const auto &peer_in_anchor : out_control_peer_in_control_anchors_[index]) {
    GE_CHECK_NOTNULL(peer_in_anchor);
    GELOGI("add control edge.src:%s, dst:%s", new_trans_nodes.back()->GetName().c_str(),
           peer_in_anchor->GetOwnerNode()->GetName().c_str());
    if (GraphUtils::AddEdge(new_trans_nodes.back()->GetOutControlAnchor(), peer_in_anchor) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                        new_trans_nodes.back()->GetName().c_str(), new_trans_nodes.back()->GetType().c_str(),
                        peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                        peer_in_anchor->GetOwnerNode()->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
             new_trans_nodes.back()->GetName().c_str(), new_trans_nodes.back()->GetType().c_str(),
             peer_in_anchor->GetOwnerNode()->GetName().c_str(), peer_in_anchor->GetOwnerNode()->GetType().c_str());
      return GRAPH_FAILED;
    }
  }

  for (const auto &peer_out_anchor : in_control_peer_out_control_anchors_[index]) {
    GE_CHECK_NOTNULL(peer_out_anchor);
    GELOGI("add control edge.src:%s, dst:%s", peer_out_anchor->GetOwnerNode()->GetName().c_str(),
           new_trans_nodes.front()->GetName().c_str());
    if (GraphUtils::AddEdge(peer_out_anchor, new_trans_nodes.front()->GetInControlAnchor()) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                        peer_out_anchor->GetOwnerNode()->GetName().c_str(),
                        peer_out_anchor->GetOwnerNode()->GetType().c_str(),
                        new_trans_nodes.front()->GetName().c_str(), new_trans_nodes.front()->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
             peer_out_anchor->GetOwnerNode()->GetName().c_str(), peer_out_anchor->GetOwnerNode()->GetType().c_str(),
             new_trans_nodes.front()->GetName().c_str(), new_trans_nodes.front()->GetType().c_str());
      return GRAPH_FAILED;
    }
  }

  for (const auto &peer_in_anchor : out_control_peer_in_data_anchors_[index]) {
    GE_CHECK_NOTNULL(peer_in_anchor);
    GELOGI("add control edge.src:%s, dst:%s", new_trans_nodes.back()->GetName().c_str(),
           peer_in_anchor->GetOwnerNode()->GetName().c_str());
    if (GraphUtils::AddEdge(new_trans_nodes.back()->GetOutControlAnchor(), peer_in_anchor) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                        new_trans_nodes.back()->GetName().c_str(), new_trans_nodes.back()->GetType().c_str(),
                        peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                        peer_in_anchor->GetOwnerNode()->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
             new_trans_nodes.back()->GetName().c_str(), new_trans_nodes.back()->GetType().c_str(),
             peer_in_anchor->GetOwnerNode()->GetName().c_str(), peer_in_anchor->GetOwnerNode()->GetType().c_str());
      return GRAPH_FAILED;
    }
  }

  for (const auto &peer_in_anchor : out_data_peer_in_control_anchors_[index]) {
    GE_CHECK_NOTNULL(peer_in_anchor);
    GELOGI("add control edge.src:%s, dst:%s", new_trans_nodes.back()->GetName().c_str(),
           peer_in_anchor->GetOwnerNode()->GetName().c_str());
    if (GraphUtils::AddEdge(new_trans_nodes.back()->GetOutDataAnchor(0), peer_in_anchor) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:%d) failed",
                        new_trans_nodes.back()->GetName().c_str(), new_trans_nodes.back()->GetType().c_str(),
                        peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                        peer_in_anchor->GetOwnerNode()->GetType().c_str(), peer_in_anchor->GetIdx());
      GELOGE(GRAPH_FAILED, "[Add][Edge] between op:%s(%s)(index:0) and op:%s(%s)(index:%d) failed",
             new_trans_nodes.back()->GetName().c_str(), new_trans_nodes.back()->GetType().c_str(),
             peer_in_anchor->GetOwnerNode()->GetName().c_str(),
             peer_in_anchor->GetOwnerNode()->GetType().c_str(), peer_in_anchor->GetIdx());
      return GRAPH_FAILED;
    }
  }

  if (sub_graph_has_out_data_peer_in_control_edge_[index]) {
    auto in_anchor = sub_graph_anchors_[index].back().second;
    GELOGI("add control edge.src:%s, dst:%s", new_trans_nodes.back()->GetName().c_str(),
           in_anchor->GetOwnerNode()->GetName().c_str());
    if (GraphUtils::AddEdge(new_trans_nodes.back()->GetOutDataAnchor(0),
                            in_anchor->GetOwnerNode()->GetInControlAnchor()) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s) and op:%s(%s) failed",
                        new_trans_nodes.back()->GetName().c_str(), new_trans_nodes.back()->GetType().c_str(),
                        in_anchor->GetOwnerNode()->GetName().c_str(), in_anchor->GetOwnerNode()->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Add][Edge] between op:%s(%s) and op:%s(%s) failed",
             new_trans_nodes.back()->GetName().c_str(), new_trans_nodes.back()->GetType().c_str(),
             in_anchor->GetOwnerNode()->GetName().c_str(), in_anchor->GetOwnerNode()->GetType().c_str());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

bool TransOpWithoutReshapeFusionPass::OpAccuracyAbilityCheck(const OpDescPtr &op_desc) {
  auto instance = GELib::GetInstance();
  if ((instance == nullptr) || (!instance->InitFlag())) {
    GELOGW("GELib is not initialized!");
    return false;
  }
  if (op_desc == nullptr) {
    return false;
  }
  OpsKernelManager &ops_kernel_manager = instance->OpsKernelManagerObj();
  vector<OpInfo> op_infos = ops_kernel_manager.GetOpsKernelInfo(op_desc->GetType());
  if (op_infos.empty()) {
    GELOGI("Can not get op info by op type:%s", op_desc->GetType().c_str());
    return false;
  }

  std::string unsupported_reason;
  for (const auto &it : op_infos) {
    auto kernel_map = ops_kernel_manager.GetAllOpsKernelInfoStores();
    auto &kernel_name = it.opKernelLib;
    auto kernel_info_store = kernel_map.find(kernel_name);
    if (kernel_info_store != kernel_map.end()) {
      if (kernel_info_store->second != nullptr &&
          kernel_info_store->second->CheckAccuracySupported(op_desc, unsupported_reason)) {
        op_desc->SetOpEngineName(it.engine);
        op_desc->SetOpKernelLibName(kernel_name);
        GELOGI("Set OpKernelLibName %s and engine name %s into op_desc %s", kernel_name.c_str(), it.engine.c_str(),
               op_desc->GetName().c_str());
        return true;
      }
    }
  }
  GELOGI("op %s CheckAccuracySupported failed!reason:%s", op_desc->GetType().c_str(), unsupported_reason.c_str());
  return false;
}

bool TransOpWithoutReshapeFusionPass::FusionFormatSupport(Format format) {
  return format == FORMAT_NCHW || format == FORMAT_NHWC || format == FORMAT_FRACTAL_Z || format == FORMAT_NC1HWC0;
}

graphStatus TransOpWithoutReshapeFusionPass::GetSubGraphsBetweenNormalNode(
  const OutDataAnchorPtr &out_anchor, std::vector<vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>>> &sub_graphs_out,
  vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> &nodes_list) {
  graphStatus ret = GRAPH_SUCCESS;
  if (out_anchor == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param out_anchor is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] param out_anchor is nullptr");
    return GRAPH_FAILED;
  }

  for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
    if (peer_in_anchor == nullptr || peer_in_anchor->GetOwnerNode() == nullptr ||
        peer_in_anchor->GetOwnerNode()->GetOpDesc() == nullptr) {
      continue;
    }

    nodes_list.emplace_back(out_anchor, peer_in_anchor);
    auto peer_in_node = peer_in_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(peer_in_node);
    if (!IsTransOp(peer_in_node)) {
      sub_graphs_out.push_back(nodes_list);
      nodes_list.pop_back();
    } else {
      for (const auto &peer_out_anchor : peer_in_node->GetAllOutDataAnchors()) {
        ret = GetSubGraphsBetweenNormalNode(peer_out_anchor, sub_graphs_out, nodes_list);
        if (ret != GRAPH_SUCCESS) {
          GELOGE(GRAPH_FAILED, "[Get][SubGraphs] Between Normal Node failed! node:%s",
                 peer_in_node->GetName().c_str());
          return GRAPH_FAILED;
        }
      }
      nodes_list.pop_back();
    }
  }
  return GRAPH_SUCCESS;
}

bool TransOpWithoutReshapeFusionPass::IsTransOp(const NodePtr &node) {
  // The caller guarantees that the pointer is not null.
  return node->GetType() == CAST || node->GetType() == RESHAPE || node->GetType() == TRANSPOSE ||
         node->GetType() == TRANSPOSED || node->GetType() == TRANSDATA;
}
}  // namespace ge
