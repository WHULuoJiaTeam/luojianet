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
#include "graph/utils/node_utils.h"
#include <stack>
#include <securec.h>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/debug/ge_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/node_impl.h"
#include "graph/ge_context.h"
#include "graph/runtime_inference_context.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/constant_utils.h"

namespace ge {
std::map<NodePtr, std::vector<uint32_t>> NodeUtils::map_send_info_{};
std::map<NodePtr, std::vector<uint32_t>> NodeUtils::map_recv_info_{};

const std::set<std::string> kConstOpTypes{ "Const", "Constant" };

const std::set<std::string> kEnterOpTypes{ "Enter", "RefEnter" };
const std::set<std::string> kMergeOpTypes{ "Merge", "RefMerge" };
const std::set<std::string> kSwitchOpTypes{ "Switch", "RefSwitch" };
const std::set<std::string> kNextIterationOpTypes{ "NextIteration", "RefNextIteration" };
const std::set<std::string> kExitOpTypes{ "Exit", "RefExit" };

const std::set<std::string> kIfOpTypes{ "If", "_If", "StatelessIf" };
const std::set<std::string> kWhileOpTypes{ "While", "_While", "StatelessWhile" };
const std::set<std::string> kCaseOpTypes{ "Case" };
const std::set<std::string> kForOpTypes{ "For" };

const char_t *const kRefIndex = "_parent_node_index";
const char_t *const kPartSrcGraph = "part_src_graph";

bool OpShapeIsUnknown(const OpDescPtr &desc) {
  for (const auto &ptr : desc->GetAllInputsDescPtr()) {
    const auto ge_shape = ptr->GetShape();
    auto dims = ge_shape.GetDims();
    if (std::any_of(dims.begin(), dims.end(),
                    [](const int64_t dim) { return ((dim == UNKNOWN_DIM) || (dim == (UNKNOWN_DIM_NUM))); })) {
      return true;
    }
  }
  for (const auto &ptr : desc->GetAllOutputsDescPtr()) {
    const auto ge_shape = ptr->GetShape();
    auto dims = ge_shape.GetDims();
    if (std::any_of(dims.begin(), dims.end(),
                    [](const int64_t dim) { return ((dim == UNKNOWN_DIM) || (dim == (UNKNOWN_DIM_NUM))); })) {
      return true;
    }
  }
  return false;
}

bool IsComputableOp(const NodePtr &node) {
  if ((node->GetType() == DATA) || (node->GetType() == NETOUTPUT)) {
    return false;
  }
  if (!node->GetOpDesc()->GetSubgraphInstanceNames().empty()) {
    return false;
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus NodeUtils::AddSendEventId(const NodePtr &node,
                                                                                     const uint32_t &event_id) {
  GE_CHECK_NOTNULL(node);
  map_send_info_[node].push_back(event_id);
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus NodeUtils::AddRecvEventId(const NodePtr &node,
                                                                                     const uint32_t &event_id) {
  GE_CHECK_NOTNULL(node);
  map_recv_info_[node].push_back(event_id);
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
NodeUtils::GetSendEventIdList(const NodePtr &node, std::vector<uint32_t> &vec_send) {
  GE_CHECK_NOTNULL(node);
  const auto find = map_send_info_.find(node);
  if (find == map_send_info_.end()) {
    return GRAPH_FAILED;
  } else {
    vec_send = find->second;
    return GRAPH_SUCCESS;
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
NodeUtils::GetRecvEventIdList(const NodePtr &node, std::vector<uint32_t> &vec_recv) {
  GE_CHECK_NOTNULL(node);
  const auto find = map_recv_info_.find(node);
  if (find == map_recv_info_.end()) {
    return GRAPH_FAILED;
  } else {
    vec_recv = find->second;
    return GRAPH_SUCCESS;
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus NodeUtils::ClearSendInfo() {
  map_send_info_.clear();
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus NodeUtils::ClearRecvInfo() {
  map_recv_info_.clear();
  return GRAPH_SUCCESS;
}

graphStatus NodeUtils::GetSingleOutputNodeOfNthLayer(const NodePtr &src, int depth, NodePtr &dst) {
  GE_CHECK_NOTNULL(src);
  NodePtr cur_ptr;
  if (depth < 1) {
    return GRAPH_FAILED;
  }
  for (int32_t i = 0; i < depth; i++) {
    if (src->GetOutDataNodes().size() != 1U) {
      return GRAPH_FAILED;
    }
    cur_ptr = src->GetOutDataNodes().at(0UL);
    GE_CHECK_NOTNULL(cur_ptr);
  }
  dst = cur_ptr;
  return GRAPH_SUCCESS;
}

graphStatus NodeUtils::GetDataOutAnchorAndControlInAnchor(const NodePtr &node_ptr, OutDataAnchorPtr &out_data,
                                                          InControlAnchorPtr &in_control) {
  GE_CHECK_NOTNULL(node_ptr);
  for (const auto &p : node_ptr->GetAllOutDataAnchors()) {
    GE_CHK_BOOL_EXEC((p != nullptr),
                     REPORT_INNER_ERROR("E19999", "GetAllOutDataAnchors is nullptr, node:%s.",
                                        node_ptr->GetName().c_str());
                     continue, "[Get][AllOutDataAnchors] is nullptr, node:%s", node_ptr->GetName().c_str());
    for (const auto &p_in : p->GetPeerInControlAnchors()) {
      GE_CHK_BOOL_EXEC((p_in != nullptr),
                       REPORT_INNER_ERROR("E19999", "GetPeerInControlAnchors is nullptr, node:%s",
                                          node_ptr->GetName().c_str());
                       continue, "[Get][PeerInDataAnchors] is nullptr, node:%s", node_ptr->GetName().c_str());
      out_data = p;
      in_control = p_in;
      return GRAPH_SUCCESS;
    }
  }
  return GRAPH_FAILED;
}

graphStatus NodeUtils::ClearInDataAnchor(const NodePtr &node_ptr, const InDataAnchorPtr &in_data_anchor) {
  GE_CHK_BOOL_EXEC((node_ptr != nullptr) && (node_ptr->impl_ != nullptr) && (in_data_anchor != nullptr),
                   REPORT_INNER_ERROR("E19999", "param node or in_data_anchor is nullptr, check invalid.");
                   return GRAPH_FAILED, "[Check][Param] node or in_data_anchor is nullptr");
  bool find_flag = false;
  uint32_t index = 0U;
  std::vector<InDataAnchorPtr>::iterator it = node_ptr->impl_->in_data_anchors_.end();
  for (const auto &tmp : node_ptr->impl_->in_data_anchors_) {
    if (tmp == in_data_anchor) {
      find_flag = true;
      const auto iter = node_ptr->impl_->in_data_anchors_.begin() + static_cast<int64_t>(index);
      if (iter != node_ptr->impl_->in_data_anchors_.end()) {
        it = node_ptr->impl_->in_data_anchors_.erase(iter);
      }
      break;
    }
    index++;
  }
  for (; it != node_ptr->impl_->in_data_anchors_.end(); ++it) {
    (*it)->SetIdx(static_cast<int32_t>(index));
    index++;
  }

  if (!find_flag) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus NodeUtils::SetAllAnchorStatus(const NodePtr &node_ptr) {
  GE_CHK_BOOL_EXEC(node_ptr != nullptr, REPORT_INNER_ERROR("E19999", "param node_ptr is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] node is nullptr");
  GE_CHK_BOOL_EXEC(SetAllAnchorStatus(*node_ptr) == GRAPH_SUCCESS,
                   REPORT_CALL_ERROR("E19999", "SetAllAnchorStatus failed, node:%s", node_ptr->GetName().c_str());
                   return GRAPH_FAILED, "[Set][AllAnchorStatus] failed, node:%s", node_ptr->GetName().c_str());
  return GRAPH_SUCCESS;
}

graphStatus NodeUtils::SetAllAnchorStatus(Node &node) {
  if (node.impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param node impl is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Node impl is nullptr.");
    return GRAPH_FAILED;
  }
  node.impl_->anchor_status_updated_ = true;
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool NodeUtils::IsAnchorStatusSet(const NodePtr &node_ptr) {
  GE_CHK_BOOL_EXEC(node_ptr != nullptr, REPORT_INNER_ERROR("E19999", "param node_ptr is nullptr, check invalid");
                   return false, "[Check][Param] node is nullptr");
  return IsAnchorStatusSet(*node_ptr);
}

bool NodeUtils::IsAnchorStatusSet(const Node &node) {
  if (node.impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param node impl is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Node impl is nullptr.");
    return false;
  }
  return node.impl_->anchor_status_updated_;
}

graphStatus NodeUtils::MoveOutputEdges(const NodePtr &origin_node, const NodePtr &new_node) {
  if ((origin_node == nullptr) || (new_node == nullptr)) {
    return GRAPH_FAILED;
  }
  auto origin_out_data_anchors = origin_node->GetAllOutDataAnchors();
  auto new_out_data_anchors = new_node->GetAllOutDataAnchors();
  if (origin_out_data_anchors.size() != new_out_data_anchors.size()) {
    return GRAPH_FAILED;
  }

  for (size_t i = 0UL; i < origin_out_data_anchors.size(); ++i) {
    for (const auto &peer_anchor : origin_out_data_anchors.at(i)->GetPeerInDataAnchors()) {
      GE_CHK_BOOL_EXEC(origin_out_data_anchors.at(i)->Unlink(peer_anchor) == GRAPH_SUCCESS,
                       REPORT_CALL_ERROR("E19999", "unlink peer_dataanchor failed, node:%s",
                                         origin_node->GetName().c_str());
                       continue, "[Unlink][PeerAnchor] failed, node:%s", origin_node->GetName().c_str());
      GE_CHK_BOOL_EXEC(new_out_data_anchors.at(i)->LinkTo(peer_anchor) == GRAPH_SUCCESS,
                       REPORT_CALL_ERROR("E19999", "LinkTo peer_dataanchor failed, node:%s",
                                         new_node->GetName().c_str());
                       continue, "[LinkTo][PeerAnchor] failed, node:%s", new_node->GetName().c_str());
    }

    for (const auto &peer_anchor : origin_out_data_anchors.at(i)->GetPeerInControlAnchors()) {
      GE_CHK_BOOL_EXEC(origin_out_data_anchors.at(i)->Unlink(peer_anchor) == GRAPH_SUCCESS,
                       REPORT_CALL_ERROR("E19999", "unlink peer_controlanchor failed, node:%s",
                                         origin_node->GetName().c_str());
                       continue, "[Unlink][PeerAnchor] failed, node:%s", origin_node->GetName().c_str());
      GE_CHK_BOOL_EXEC(new_out_data_anchors.at(i)->LinkTo(peer_anchor) == GRAPH_SUCCESS,
                       REPORT_CALL_ERROR("E19999", "LinkTo peer_controlanchor failed, node:%s",
                                         new_node->GetName().c_str());
                       continue, "[LinkTo][PeerAnchor] failed, node:%s", new_node->GetName().c_str());
    }
  }

  const auto origin_out_control_anchor = origin_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(origin_out_control_anchor);
  const auto new_out_control_anchor = new_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(new_out_control_anchor);
  for (const auto &peer_anchor : origin_out_control_anchor->GetPeerInControlAnchors()) {
    GE_CHK_BOOL_EXEC(new_out_control_anchor->LinkTo(peer_anchor) == GRAPH_SUCCESS,
                     REPORT_CALL_ERROR("E19999", "linkto peer_anchor from %s to %s failed.",
                                       new_out_control_anchor->GetOwnerNode()->GetName().c_str(),
                                       peer_anchor->GetOwnerNode()->GetName().c_str());
                     continue, "[LinkTo][PeerAnchor] from %s to %s failed",
                     new_out_control_anchor->GetOwnerNode()->GetName().c_str(),
                     peer_anchor->GetOwnerNode()->GetName().c_str());
  }
  for (const auto &peer_anchor : origin_out_control_anchor->GetPeerInDataAnchors()) {
    GE_CHK_BOOL_EXEC(new_out_control_anchor->LinkTo(peer_anchor) == GRAPH_SUCCESS,
                     REPORT_CALL_ERROR("E19999", "linkto peer_anchor from %s to %s failed.",
                                       new_out_control_anchor->GetOwnerNode()->GetName().c_str(),
                                       peer_anchor->GetOwnerNode()->GetName().c_str());
                     continue, "[LinkTo][PeerAnchor] from %s to %s failed",
                     new_out_control_anchor->GetOwnerNode()->GetName().c_str(),
                     peer_anchor->GetOwnerNode()->GetName().c_str());
  }
  origin_out_control_anchor->UnlinkAll();

  return GRAPH_SUCCESS;
}

bool NodeUtils::IsConst(const Node &node) {
  const auto src_node_type = node.GetType();
  const bool is_const = ((src_node_type == CONSTANT) || (src_node_type == CONSTANTOP));
  return is_const;
}

void NodeUtils::UpdateIsInputConst(const NodePtr &node_ptr) {
  if (node_ptr == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node_ptr is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] node is null");
    return;
  }
  UpdateIsInputConst(*node_ptr);
}

///
/// update is_input_const
/// @param node
/// @return void
///
void NodeUtils::UpdateIsInputConst(Node &node) {
  std::vector<bool> is_input_const;
  const size_t anchor_num = node.GetAllInDataAnchors().size();
  for (size_t i = 0UL; i < anchor_num; i++) {
    const auto in_anchor = node.GetInDataAnchor(static_cast<int32_t>(i));
    if (in_anchor == nullptr) {
      is_input_const.push_back(false);
      continue;
    }
    const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      is_input_const.push_back(false);
      continue;
    }
    const auto src_node = peer_out_anchor->GetOwnerNode();
    if (src_node == nullptr) {
      is_input_const.push_back(false);
      continue;
    }
    if (IsConst(*(src_node))) {
      is_input_const.push_back(true);
    } else {
      is_input_const.push_back(false);
    }
  }
  if (node.GetOpDesc() == nullptr) {
    REPORT_INNER_ERROR("E19999", "node has no opdesc.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Node get opdesc is nullptr");
    return;
  }
  node.GetOpDesc()->SetIsInputConst(is_input_const);
}

void NodeUtils::UnlinkAll(const Node &node) {
  for (const auto &anchor : node.GetAllOutAnchors()) {
    anchor->UnlinkAll();
  }
  for (const auto &anchor : node.GetAllInAnchors()) {
    anchor->UnlinkAll();
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus NodeUtils::UpdatePeerNodeInputDesc(const NodePtr &node_ptr) {
  if (node_ptr == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node_ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Nodeptr is nullptr");
    return GRAPH_FAILED;
  }
  auto op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    return GRAPH_FAILED;
  }
  bool is_unknown_graph = node_ptr->GetOwnerComputeGraph()->GetGraphUnknownFlag();
  if (is_unknown_graph) {
    return GRAPH_SUCCESS;
  }
  for (const auto &out_anchor : node_ptr->GetAllOutDataAnchors()) {
    auto output_tensor = op_desc->MutableOutputDesc(out_anchor->GetIdx());
    auto out_dims = output_tensor->GetShape().GetDims();
    auto out_dtype = output_tensor->GetDataType();

    GELOGD("node name is %s, origin shape is %ld, origin format is %s, origin data type is %s",
           node_ptr->GetName().c_str(), output_tensor->GetOriginShape().GetShapeSize(),
           TypeUtils::FormatToSerialString(output_tensor->GetOriginFormat()).c_str(),
           TypeUtils::DataTypeToSerialString(output_tensor->GetOriginDataType()).c_str());

    for (const auto &peer_anchor : out_anchor->GetPeerInDataAnchors()) {
      auto peer_anchor_opdesc = peer_anchor->GetOwnerNode()->GetOpDesc();
      if (peer_anchor_opdesc == nullptr) {
        REPORT_INNER_ERROR("E19999", "peer data anchor ownernode:%s get op desc return nullptr.",
                           peer_anchor->GetOwnerNode()->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Invoke][GetOpDesc]peer data anchor ownernode:%s get op desc return nullptr.",
               peer_anchor->GetOwnerNode()->GetName().c_str());
        continue;
      }
      if (op_desc->GetId() < peer_anchor_opdesc->GetId() ||
          peer_anchor_opdesc->GetType() == CONSTANT ||
          peer_anchor_opdesc->GetType() == CONSTANTOP) {
          GELOGD("no need to UpdatePeerNodeInputDesc");
          continue;
      }
      auto peer_input_desc = peer_anchor->GetOwnerNode()->GetOpDesc()->MutableInputDesc(peer_anchor->GetIdx());
      if (peer_input_desc == nullptr) {
        REPORT_INNER_ERROR("E19999", "node:%s out anchor to in anchor(%d)'s input desc is nullptr",
                           peer_anchor->GetOwnerNode()->GetName().c_str(), peer_anchor->GetIdx());
        GELOGE(GRAPH_FAILED, "[Invoke][MutableInputDesc] node:%s out anchor to in anchor(%d)'s input desc is nullptr",
               peer_anchor->GetOwnerNode()->GetName().c_str(), peer_anchor->GetIdx());
        continue;
      }
      // check shape and dtype continuity. do not stop process
      auto peer_input_dims = peer_input_desc->GetShape().GetDims();
      auto peer_input_dtype = peer_input_desc->GetDataType();
      if (out_dtype != peer_input_dtype) {
        GELOGW("[Update][PeerInput] current node [%s] [%d]\'th out_dtype is [%s].peer input node [%s] [%d]\'th "
               "input_dtype is [%s].The two dtype should be same! Please check graph and fix it",
               node_ptr->GetName().c_str(), out_anchor->GetIdx(), TypeUtils::DataTypeToSerialString(out_dtype).c_str(),
               peer_anchor->GetOwnerNode()->GetName().c_str(), peer_anchor->GetIdx(),
               TypeUtils::DataTypeToSerialString(peer_input_dtype).c_str());
      } else if ((!peer_input_dims.empty()) && (out_dims != peer_input_dims)) {
        GELOGW("[Update][PeerInput] current node [%s] [%d]\'th out_shape is [%s].peer input node [%s] [%d]\'th "
               "input_shape is [%s].The two shape should be same! Please check graph and fix it",
               node_ptr->GetName().c_str(), out_anchor->GetIdx(), output_tensor->GetShape().ToString().c_str(),
               peer_anchor->GetOwnerNode()->GetName().c_str(), peer_anchor->GetIdx(),
               peer_input_desc->GetShape().ToString().c_str());
      }
      GELOGI("Peer input opdesc name is %s, need to flush: shape size is %zu, datatype is %d, original datatype is %d",
             peer_anchor->GetOwnerNode()->GetOpDesc()->GetName().c_str(),
             output_tensor->GetShape().GetDimNum(), output_tensor->GetDataType(),
             output_tensor->GetOriginDataType());
      peer_input_desc->SetOriginShape(output_tensor->GetOriginShape());
      peer_input_desc->SetShape(output_tensor->GetShape());
      peer_input_desc->SetDataType(output_tensor->GetDataType());
      peer_input_desc->SetOriginDataType(output_tensor->GetOriginDataType());
      std::vector<std::pair<int64_t, int64_t>> shape_range;
      (void) output_tensor->GetShapeRange(shape_range);
      peer_input_desc->SetShapeRange(shape_range);
      ge::TensorUtils::SetRealDimCnt(*peer_input_desc,
                                     static_cast<uint32_t>(output_tensor->GetShape().GetDims().size()));
      GELOGI("Peer input opdesc name is %s, shape size is %zu, datatype is %d, original datatype is %d",
             peer_anchor->GetOwnerNode()->GetOpDesc()->GetName().c_str(),
             peer_input_desc->GetShape().GetDimNum(), peer_input_desc->GetDataType(),
             peer_input_desc->GetOriginDataType());
    }
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus NodeUtils::AppendInputAnchor(const NodePtr &node, uint32_t num) {
  if (node == nullptr || node->impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Input node is null");
    return GRAPH_FAILED;
  }

  const GeTensorDesc data_desc(GeShape(), FORMAT_ND, DT_FLOAT);
  const auto &op_desc = node->GetOpDesc();
  for (size_t i = op_desc->GetInputsSize(); i < num; ++i) {
    if (op_desc->AddInputDesc(data_desc) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "AddInputDesc failed, op:%s", op_desc->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Add][InputDesc] failed, op:%s", op_desc->GetName().c_str());
      return GRAPH_FAILED;
    }
  }

  for (size_t i = node->impl_->in_data_anchors_.size(); i < num; ++i) {
    const auto anchor = ComGraphMakeShared<InDataAnchor>(node, i);
    if (anchor == nullptr) {
      REPORT_CALL_ERROR("E19999", "Current in data anchor is null, make shared_ptr failed.");
      GELOGE(OUT_OF_MEMORY, "[Create][InDataAnchor] Current in data anchor is null, make shared_ptr failed.");
      return GRAPH_FAILED;
    }
    node->impl_->in_data_anchors_.push_back(anchor);
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus NodeUtils::RemoveInputAnchor(const NodePtr &node, uint32_t num) {
  if (node == nullptr || node->impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node is null, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Input node is null");
    return GRAPH_FAILED;
  }

  const auto &op_desc = node->GetOpDesc();
  while (op_desc->GetInputsSize() > num) {
    if (!OpDescUtils::ClearInputDesc(op_desc, num)) {
      return GRAPH_FAILED;
    }
  }

  const auto input_names = op_desc->GetAllInputName();
  (void)op_desc->UpdateInputName(input_names);
  auto is_input_const = op_desc->GetIsInputConst();
  is_input_const.resize(static_cast<std::size_t>(num));
  op_desc->SetIsInputConst(is_input_const);

  while (node->impl_->in_data_anchors_.size() > num) {
    node->impl_->in_data_anchors_.pop_back();
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus NodeUtils::AppendOutputAnchor(const NodePtr &node, uint32_t num) {
  if (node == nullptr || node->impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "Input node is null, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Input node is null");
    return GRAPH_FAILED;
  }

  const GeTensorDesc data_desc(GeShape(), FORMAT_ND, DT_FLOAT);
  const OpDescPtr &op_desc = node->GetOpDesc();
  for (size_t i = op_desc->GetOutputsSize(); i < num; ++i) {
    if (op_desc->AddOutputDesc(data_desc) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add output desc failed, op:%s", op_desc->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Add][OutputDesc] failed, op:%s", op_desc->GetName().c_str());
      return GRAPH_FAILED;
    }
  }

  for (size_t i = node->impl_->out_data_anchors_.size(); i < num; ++i) {
    const auto anchor = ComGraphMakeShared<OutDataAnchor>(node, i);
    if (anchor == nullptr) {
      REPORT_CALL_ERROR("E19999", "Current out data anchor is null, make shared_ptr failed.");
      GELOGE(OUT_OF_MEMORY, "[Create][OutDataAnchor] Current out data anchor is null, make shared_ptr failed.");
      return GRAPH_FAILED;
    }
    node->impl_->out_data_anchors_.push_back(anchor);
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus NodeUtils::RemoveOutputAnchor(const NodePtr &node, uint32_t num) {
  if (node == nullptr || node->impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "Input node is null, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Input node is null");
    return GRAPH_FAILED;
  }

  const auto &op_desc = node->GetOpDesc();
  const auto output_names = op_desc->GetAllOutputName();
  while (op_desc->GetOutputsSize() > num) {
    if (!OpDescUtils::ClearOutputDesc(op_desc, num)) {
      return GRAPH_FAILED;
    }
  }
  (void)op_desc->UpdateOutputName(output_names);

  while (node->impl_->out_data_anchors_.size() > num) {
    node->impl_->out_data_anchors_.pop_back();
  }

  return GRAPH_SUCCESS;
}

bool NodeUtils::IsInNodesEmpty(const Node &node) {
  if (node.impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "Node impl is null, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Node impl is null");
    return false;
  }
  for (const auto &in_anchor : node.impl_->in_data_anchors_) {
    if (in_anchor != nullptr) {
      const auto out_anchor = in_anchor->GetPeerOutAnchor();
      if (out_anchor != nullptr) {
        if (out_anchor->GetOwnerNode() != nullptr) {
          return false;
        }
      }
    }
  }

  if ((node.impl_->in_control_anchor_ != nullptr) &&
      (!node.impl_->in_control_anchor_->IsPeerOutAnchorsEmpty())) {
    const auto peer_out_control_anchors = node.impl_->in_control_anchor_->GetPeerOutControlAnchors();
    for (const auto &out_control_anchor : peer_out_control_anchors) {
      if (out_control_anchor != nullptr) {
        if (out_control_anchor->GetOwnerNode() != nullptr) {
          return false;
        }
      }
    }
  }

  return true;
}
GeTensorDesc NodeUtils::GetOutputDesc(const Node &node, uint32_t index) {
  const auto desc = node.GetOpDesc();
  if (desc == nullptr) {
    return GeTensorDesc();
  }
  return desc->GetOutputDesc(index);
}
GeTensorDesc NodeUtils::GetInputDesc(const Node &node, uint32_t index) {
  const auto desc = node.GetOpDesc();
  if (desc == nullptr) {
    return GeTensorDesc();
  }
  return desc->GetInputDesc(index);
}
graphStatus NodeUtils::UpdateOutputShape(const Node &node, uint32_t index, const GeShape &shape) {
  const auto desc = node.GetOpDesc();
  if (desc == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  const auto output_desc = desc->MutableOutputDesc(index);
  if (output_desc == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  output_desc->SetShape(shape);
  return GRAPH_SUCCESS;
}
graphStatus NodeUtils::UpdateInputShape(const Node &node, uint32_t index, const GeShape &shape) {
  const auto desc = node.GetOpDesc();
  if (desc == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  const auto input_desc = desc->MutableInputDesc(index);
  if (input_desc == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  input_desc->SetShape(shape);
  return GRAPH_SUCCESS;
}

graphStatus NodeUtils::GetNodeUnknownShapeStatus(const Node &node, bool &is_unknow) {
  const auto desc = node.GetOpDesc();
  GE_CHECK_NOTNULL(desc);
  // check self
  is_unknow = OpShapeIsUnknown(desc);
  if (is_unknow) {
    return GRAPH_SUCCESS;
  }
  const auto sub_graph_names = desc->GetSubgraphInstanceNames();
  if (sub_graph_names.empty()) {
    return GRAPH_SUCCESS;
  } else {
    const auto owner_graph = node.GetOwnerComputeGraph();
    GE_CHECK_NOTNULL(owner_graph);
    // During graph splitting, get parent graph cannot be obtained in some scenarios,
    // but the root graph can be set use the attribute.
    ge::ComputeGraphPtr src_graph = owner_graph->TryGetExtAttr(kPartSrcGraph, ge::ComputeGraphPtr());
    if (src_graph == nullptr) {
      GELOGD("src graph is null, owner graph name is %s", owner_graph->GetName().c_str());
      src_graph = owner_graph;
    }
    GELOGD("src graph is %s, owner graph name is %s", src_graph->GetName().c_str(), owner_graph->GetName().c_str());
    const auto root_graph = GraphUtils::FindRootGraph(src_graph);
    if (root_graph == nullptr) {
      REPORT_INNER_ERROR("E19999", "node:%s has no root graph.", node.GetName().c_str());
      GE_LOGE("[Get][Graph] Node %s gets null root graph", node.GetName().c_str());
      return GRAPH_PARAM_INVALID;
    }
    for (auto &sub_graph_name : sub_graph_names) {
      const auto sub_graph = root_graph->GetSubgraph(sub_graph_name);
      GE_CHECK_NOTNULL(sub_graph);
      for (const auto &node_ptr : sub_graph->GetDirectNode()) {
        const auto status = GetNodeUnknownShapeStatus(*node_ptr, is_unknow);
        if (status != GRAPH_SUCCESS) {
          REPORT_CALL_ERROR("E19999", "GetNodeUnknownShapeStatus failed, node:%s, status:%d",
                            node_ptr->GetName().c_str(), status);
          GE_LOGE("[Get][NodeUnknownShapeStatus] failed! node:%s, status:%d", node_ptr->GetName().c_str(), status);
          return status;
        }
        if (is_unknow) {
          return GRAPH_SUCCESS;
        }
      }
    }
  }
  return GRAPH_SUCCESS;
}

std::string NodeUtils::GetNodeType(const Node &node) {
  if (node.GetType() != FRAMEWORKOP) {
    return node.GetType();
  }

  std::string type;
  (void)AttrUtils::GetStr(node.GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, type);
  return type;
}

std::string NodeUtils::GetNodeType(const NodePtr &node) {
  return node == nullptr ? "" : GetNodeType(*node);
}

std::vector<ComputeGraphPtr> NodeUtils::GetAllSubgraphs(const Node &node) {
  const auto op_desc = node.GetOpDesc();
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "Failed to get op desc from node %s ", node.GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] Failed to get op desc from node %s ", node.GetName().c_str());
    return {};
  }
  const auto root_graph = GraphUtils::FindRootGraph(node.GetOwnerComputeGraph());
  if (root_graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Failed to find root graph from node %s ", node.GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][Graph] Failed to find root graph from node %s ", node.GetName().c_str());
    return {};
  }
  return root_graph->GetAllSubgraphs();
}

graphStatus NodeUtils::GetDirectSubgraphs(const NodePtr &node, std::vector<ComputeGraphPtr> &subgraphs) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    REPORT_INNER_ERROR("E19999", "node or op_desc is null");
    GELOGE(GRAPH_FAILED, "[Check][Param] node or op_desc is null");
    return GRAPH_FAILED;
  }

  const auto &root_graph = GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());
  if (root_graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Failed to find root graph from node %s ", node->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][Graph] Failed to find root graph from node %s ", node->GetName().c_str());
    return GRAPH_FAILED;
  }

  for (const auto &graph_name : node->GetOpDesc()->GetSubgraphInstanceNames()) {
    const auto &graph = root_graph->GetSubgraph(graph_name);
    if (graph == nullptr) {
      GELOGW("[Get][Subgraph] subgraph %s of node %s is null", graph_name.c_str(), node->GetName().c_str());
      continue;
    }
    subgraphs.emplace_back(graph);
  }

  return GRAPH_SUCCESS;
}

ComputeGraphPtr NodeUtils::GetSubgraph(const Node &node, uint32_t index) {
  const auto op_desc = node.GetOpDesc();
  if (op_desc == nullptr) {
    return nullptr;
  }
  const auto root_graph = GraphUtils::FindRootGraph(node.GetOwnerComputeGraph());
  if (root_graph == nullptr) {
    return nullptr;
  }
  return root_graph->GetSubgraph(op_desc->GetSubgraphInstanceName(index));
}

graphStatus NodeUtils::SetSubgraph(Node &node, uint32_t index, const ComputeGraphPtr &subgraph) {
  if (subgraph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Failed to set subgraph to node %s index %u, null subgraph",
                       node.GetName().c_str(), index);
    GE_LOGE("[Check][Param] Failed to set subgraph to node %s index %u, null subgraph", node.GetName().c_str(), index);
    return GRAPH_PARAM_INVALID;
  }
  const auto op_desc = node.GetOpDesc();
  if (op_desc == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  const auto root_graph = GraphUtils::FindRootGraph(node.GetOwnerComputeGraph());
  if (root_graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Failed to add subgraph to node %s, null root graph", node.GetName().c_str());
    GE_LOGE("[Get][Graph] Failed to add subgraph to node %s, null root graph", node.GetName().c_str());
    return GRAPH_PARAM_INVALID;
  }
  const auto ret = op_desc->SetSubgraphInstanceName(index, subgraph->GetName());
  if (ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Failed to set subgraph to node %s index %u", node.GetName().c_str(), index);
    GE_LOGE("[Set][Name] Failed to set subgraph to node %s index %u", node.GetName().c_str(), index);
    return ret;
  }
  subgraph->SetParentNode(node.shared_from_this());
  subgraph->SetParentGraph(node.GetOwnerComputeGraph());
  return root_graph->AddSubgraph(subgraph);
}

///
/// Check if node is input of subgraph
/// @param [in] node
/// @return bool
///
bool NodeUtils::IsSubgraphInput(const NodePtr &node) {
  if ((node == nullptr) || (node->GetOpDesc() == nullptr) ||
      (node->GetOwnerComputeGraph()->GetParentNode() == nullptr)) {
    return false;
  }

  const auto parent_op_desc = node->GetOwnerComputeGraph()->GetParentNode()->GetOpDesc();
  if (parent_op_desc == nullptr) {
    return false;
  }

  // dynamic shape unknown graph false
  // dynamic shape known graph with functional subgraph maybe true
  if (AttrUtils::HasAttr(parent_op_desc, ATTR_NAME_IS_UNKNOWN_SHAPE)) {
    if (node->GetOwnerComputeGraph()->GetParentGraph()->GetGraphUnknownFlag()) {
      return false;
    } else {
      if (node->GetOwnerComputeGraph()->GetParentNode()->GetOwnerComputeGraph()->GetParentNode() == nullptr) {
        return false;
      }
    }
  }

  return node->GetOpDesc()->HasAttr(ATTR_NAME_PARENT_NODE_INDEX);
}

///
/// Check if node is output of subgraph
/// @param [in] node
/// @return bool
///
bool NodeUtils::IsSubgraphOutput(const NodePtr &node) {
  if ((node == nullptr) || (node->GetOpDesc() == nullptr) ||
      (node->GetOwnerComputeGraph()->GetParentNode() == nullptr) || (node->GetType() != NETOUTPUT)) {
    return false;
  }

  const auto parent_op_desc = node->GetOwnerComputeGraph()->GetParentNode()->GetOpDesc();
  if (parent_op_desc == nullptr) {
    return false;
  }

  if (AttrUtils::HasAttr(parent_op_desc, ATTR_NAME_IS_UNKNOWN_SHAPE)) {
    if (node->GetOwnerComputeGraph()->GetParentGraph()->GetGraphUnknownFlag()) {
      return false;
    } else {
      if (node->GetOwnerComputeGraph()->GetParentNode()->GetOwnerComputeGraph()->GetParentNode() == nullptr) {
        return false;
      }
    }
  }

  for (GeTensorDesc &tensor : node->GetOpDesc()->GetAllInputsDesc()) {
    if (AttrUtils::HasAttr(tensor, ATTR_NAME_PARENT_NODE_INDEX)) {
      return true;
    }
  }

  return false;
}

///
/// @brief Get subgraph original input node.
/// @param [in] node
/// @return Node
///
NodePtr NodeUtils::GetParentInput(const Node &node) {
  uint32_t parent_index = 0U;
  if (!AttrUtils::GetInt(node.GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    return nullptr;
  }

  // Subgraph Data Node, check for constant input.
  const ComputeGraphPtr &graph = node.GetOwnerComputeGraph();
  GE_CHECK_NOTNULL_EXEC(graph, return nullptr);

  const NodePtr &parent_node = graph->GetParentNode();
  GE_CHECK_NOTNULL_EXEC(parent_node, return nullptr);

  const InDataAnchorPtr &in_anchor = parent_node->GetInDataAnchor(static_cast<int32_t>(parent_index));
  GE_CHECK_NOTNULL_EXEC(in_anchor, return nullptr);

  const OutDataAnchorPtr &peer_out_anchor = in_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL_EXEC(peer_out_anchor, return nullptr);

  return peer_out_anchor->GetOwnerNode();
}

NodePtr NodeUtils::GetParentInput(const NodePtr &node) {
  return node == nullptr ? node : GetParentInput(*node);
}
NodeToOutAnchor NodeUtils::GetParentInputAndAnchor(const NodePtr &node) {
  uint32_t parent_index = 0U;
  if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    return {nullptr, nullptr};
  }

  // Subgraph Data Node, check for constant input.
  const ComputeGraphPtr &graph = node->GetOwnerComputeGraph();
  if (graph == nullptr) {
    return {nullptr, nullptr};
  }

  const NodePtr &parent_node = graph->GetParentNode();
  if (parent_node == nullptr) {
    return {nullptr, nullptr};
  }

  const InDataAnchorPtr &in_anchor = parent_node->GetInDataAnchor(static_cast<int32_t>(parent_index));
  if (in_anchor == nullptr) {
    return {nullptr, nullptr};
  }

  const OutDataAnchorPtr &peer_out_anchor = in_anchor->GetPeerOutAnchor();
  if (peer_out_anchor == nullptr) {
    return {nullptr, nullptr};
  }

  return std::make_pair(peer_out_anchor->GetOwnerNode(), peer_out_anchor);
}

///
/// @brief Get is dynamic shape graph from node.
/// @param [in] node
/// @return bool
///
bool NodeUtils::IsDynamicShape(const Node &node) {
  const auto graph = GraphUtils::FindRootGraph(node.GetOwnerComputeGraph());
  if (graph == nullptr) {
    return false;
  }

  bool is_dynamic_shape = false;
  (void)AttrUtils::GetBool(graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape);
  return is_dynamic_shape;
}

bool NodeUtils::IsDynamicShape(const NodePtr &node) {
  return node == nullptr ? false : IsDynamicShape(*node);
}

///
/// @brief Check is varying_input for while node
/// @param [in] node: Data node for subgraph
/// @return bool
///
bool NodeUtils::IsWhileVaryingInput(const ge::NodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  if (node->GetType() != DATA) {
    return false; // not input_node for subgraph
  }

  const NodePtr &parent_node = node->GetOwnerComputeGraph()->GetParentNode();
  if (parent_node == nullptr) {
    return false; // root graph
  }

  if (kWhileOpTypes.count(parent_node->GetType()) == 0U) {
    return false; // not input_node for while subgraph
  }

  uint32_t index_i = 0U;
  if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, index_i)) {
    GELOGW("[Check][Attr] Node %s has no attr PARENT_NODE_INDEX.", node->GetName().c_str());
    return false;
  }
  bool varying_flag = true;
  for (const auto &item : node->GetOutDataNodesAndAnchors()) {
    if (item.first->GetType() != NETOUTPUT) {
      continue;
    }
    const OpDescPtr op_desc = item.first->GetOpDesc();
    uint32_t index_o = 0U;
    if ((op_desc == nullptr) ||
        (!AttrUtils::GetInt(op_desc->GetInputDesc(static_cast<uint32_t>(item.second->GetIdx())),
                            ATTR_NAME_PARENT_NODE_INDEX, index_o))) {
      continue; // input for while-cond subgraph
    }
    if (index_i != index_o) {
      continue; // varying input for while-body subgraph
    }
    varying_flag = false;
    break;
  }
  return varying_flag;
}

///
/// @brief Get subgraph input is constant.
/// @param [in] node
/// @param [out] string
/// @return bool
///
bool NodeUtils::GetConstOpType(const NodePtr &node, std::string &type) {
  if (node == nullptr) {
    return false;
  }

  const auto node_type = node->GetType();
  if ((node_type == CONSTANT) || (node_type == CONSTANTOP) || (node_type == FILECONSTANT)) {
    type = node->GetType();
    return true;
  }

  if (node_type != DATA) {
    return false;   // not subgraph input node
  }

  const auto &parent = GetParentInput(node);
  return GetConstOpType(parent, type);
}

///
/// @brief Remove node-related subgraphs, including subgraphs of nodes in the subgraph.
/// @param [in] node
/// @return return GRAPH_SUCCESS if remove successfully, other for failed.
///
graphStatus NodeUtils::RemoveSubgraphsOnNode(const NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  const auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  const auto subgraph_names = op_desc->GetSubgraphInstanceNames();
  if (subgraph_names.empty()) {
    return GRAPH_SUCCESS;
  } else {
    const auto owner_graph = node->GetOwnerComputeGraph();
    GE_CHECK_NOTNULL(owner_graph);
    const auto root_graph = GraphUtils::FindRootGraph(owner_graph);
    GE_CHECK_NOTNULL(root_graph);

    std::set<std::string> subgraph_to_remove;
    for (auto &subgraph_name : subgraph_names) {
      std::deque<std::string> queue;
      queue.push_back(subgraph_name);
      (void)subgraph_to_remove.insert(subgraph_name);
      op_desc->RemoveSubgraphInstanceName(subgraph_name);
      while (!queue.empty()) {
        const auto graph_name = queue.front();
        queue.pop_front();

        const auto subgraph = root_graph->GetSubgraph(graph_name);
        GE_CHECK_NOTNULL(subgraph);
        for (const auto &sub_node : subgraph->GetDirectNode()) {
          const auto sub_op_desc = sub_node->GetOpDesc();
          GE_CHECK_NOTNULL(sub_op_desc);
          const auto sub_names = sub_op_desc->GetSubgraphInstanceNames();
          // Subgraph and all nodes in it will be removed later,
          // no need to remove 'SubgraphInstanceName' in op desc here.
          for (auto &name : sub_names) {
            if (subgraph_to_remove.insert(name).second) {
              queue.push_back(name);
            }
          }
        }
      }
    }
    // Remove subgraph from root_graph
    for (const auto &name : subgraph_to_remove) {
      GELOGI("Remove subgraph:%s.", name.c_str());
      root_graph->RemoveSubgraph(name);
    }
  }

  return GRAPH_SUCCESS;
}
///
/// @brief Get subgraph input data node by index.
/// @param [in] node
/// @return Node
///
std::vector<NodePtr> NodeUtils::GetSubgraphDataNodesByIndex(const Node &node, int index) {
  std::vector<NodePtr> in_data_node_vec;
  const auto op_desc = node.GetOpDesc();
  GE_CHECK_NOTNULL_EXEC(op_desc, return in_data_node_vec);
  const auto subgraph_names = op_desc->GetSubgraphInstanceNames();
  if (subgraph_names.empty()) {
    return in_data_node_vec;
  }
  const auto compute_graph = node.GetOwnerComputeGraph();
  for (const std::string &instance_name : subgraph_names) {
    const auto subgraph = compute_graph->GetSubgraph(instance_name);
    for (const auto &node_in_subgraph : subgraph->GetDirectNode()) {
      if (NodeUtils::IsSubgraphInput(node_in_subgraph)) {
        int32_t parent_index = -1;
        (void)AttrUtils::GetInt(node_in_subgraph->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index);
        if (parent_index == index) {
          in_data_node_vec.emplace_back(node_in_subgraph);
        }
      }
    }
  }
  return in_data_node_vec;
}
///
/// @brief Get subgraph input data node by index.
/// @param [in] node
/// @return Node
///
std::vector<NodePtr> NodeUtils::GetSubgraphOutputNodes(const Node &node) {
  std::vector<NodePtr> out_data_node_vec;
  const auto op_desc = node.GetOpDesc();
  GE_CHECK_NOTNULL_EXEC(op_desc, return out_data_node_vec);
  const auto subgraph_names = op_desc->GetSubgraphInstanceNames();
  if (subgraph_names.empty()) {
    GELOGI("Node %s is single node without sub graph.", node.GetName().c_str());
    return out_data_node_vec;
  }
  const auto compute_graph = node.GetOwnerComputeGraph();
  for (const std::string &instance_name : subgraph_names) {
    const auto subgraph = compute_graph->GetSubgraph(instance_name);
    if (subgraph == nullptr) {
      continue;
    }
    for (const auto &node_in_subgraph : subgraph->GetDirectNode()) {
      if (NodeUtils::IsSubgraphOutput(node_in_subgraph)) {
        out_data_node_vec.emplace_back(node_in_subgraph);
      }
    }
  }
  return out_data_node_vec;
}

NodePtr NodeUtils::GetInDataNodeByIndex(const Node &node, const int index) {
  if (node.GetInDataAnchor(index) == nullptr) {
    return nullptr;
  }
  if (node.GetInDataAnchor(index)->GetPeerOutAnchor() == nullptr) {
    return nullptr;
  }
  return node.GetInDataAnchor(index)->GetPeerOutAnchor()->GetOwnerNode();
}

std::vector<std::pair<InDataAnchorPtr, NodePtr>> NodeUtils::GetOutDataNodesWithAnchorByIndex(const Node &node,
                                                                                             const int index) {
  std::vector<std::pair<InDataAnchorPtr, NodePtr>> out_data_nodes;
  const auto out_data_anchor = node.GetOutDataAnchor(index);
  if (out_data_anchor == nullptr) {
    return out_data_nodes;
  }

  for (const auto peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    if (peer_in_anchor == nullptr) {
      continue;
    }
    if (peer_in_anchor->GetOwnerNode() == nullptr) {
      continue;
    }
    out_data_nodes.emplace_back(std::make_pair(peer_in_anchor, peer_in_anchor->GetOwnerNode()));
  }
  return out_data_nodes;
}

ConstNodePtr NodeUtils::GetNodeFromOperator(const Operator &oprt) {
  return oprt.GetNode();
}

std::string NodeUtils::GetInConstNodeTypeCrossSubgraph(const NodePtr &node) {
  const NodePtr input_node = GetInNodeCrossSubgraph(node);
  if (input_node == nullptr) {
    return "";
  }

  return input_node->GetType();
}

NodePtr NodeUtils::GetInNodeCrossSubgraph(const NodePtr &node) {
  NodePtr input_node = node;
  while (input_node != nullptr) {
    if (input_node->GetType() != DATA) {
      return input_node;
    }

    const auto owner_graph = input_node->GetOwnerComputeGraph();
    const auto parent_node = owner_graph->GetParentNode();
    if ((parent_node == nullptr) || (kWhileOpTypes.count(parent_node->GetType()) > 0UL)) {
      return node;       // not in subgraph or while subgraph.
    }

    input_node = GetParentInput(input_node);
  }

  return input_node;
}

NodePtr NodeUtils::CreatNodeWithoutGraph(const OpDescPtr op_desc) {
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "The OpDesc ptr should not be null.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The OpDesc ptr should not be null.");
    return nullptr;
  }
  const NodePtr node_ptr = shared_ptr<Node>(new (std::nothrow) Node(op_desc, nullptr));
  if (node_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "create node failed.");
    GELOGE(GRAPH_FAILED, "[Create][Node] node_ptr is NULL!");
    return nullptr;
  }
  return node_ptr;
}

graphStatus NodeUtils::GetInNodeCrossPartionedCallNode(const NodePtr &node, uint32_t index, NodePtr &peer_node) {
  GE_CHECK_NOTNULL(node);
  if ((node->GetAllInDataAnchorsSize() <= index) && (node->GetType() != DATA)) {
    return GRAPH_FAILED;
  }
  GELOGD("in node:%s index:%d", node->GetName().c_str(), index);
  peer_node = (node->GetType() == DATA) ? node : GetInDataNodeByIndex(*node, index);
  int32_t peer_out_anchor_index = -1;
  if (peer_node == nullptr) {
    // A->B
    // Asuming A and B belongs to different engine, during graph partition, A will be set to B's extra attr as
    // parent node. when FE get parent node A from B, check A's in_anchor peer_out_anchor is null.
    return GRAPH_SUCCESS;
  }
  while (!IsComputableOp(peer_node)) {
    if (peer_node->GetType() == DATA) {
      auto parent_node_2_anchor = GetParentInputAndAnchor(peer_node);
      if ((parent_node_2_anchor.first == nullptr) && (parent_node_2_anchor.second == nullptr)) {
        GELOGW("Returned peer_out_node is nullptr because no attr[%s] on DATA[%s] node!",
               kRefIndex, peer_node->GetName().c_str());
        peer_node = nullptr;
        return GRAPH_SUCCESS;
      }
      peer_node = parent_node_2_anchor.first;
      peer_out_anchor_index = parent_node_2_anchor.second->GetIdx();
      continue;
    }

    if (peer_node->GetType() != PARTITIONEDCALL) {
      if (peer_node->GetOpDesc()->GetSubgraphInstanceNames().empty()) {
        GELOGI("Node [%s] type [%s], real peer in node [%s] type[%s].", node->GetName().c_str(),
               node->GetType().c_str(), peer_node->GetName().c_str(), peer_node->GetType().c_str());
        return GRAPH_SUCCESS;
      }
      // other subgraph(if,while,case) currently not support, return node and warn
      GELOGW("Node [%s] type [%s], real peer in node [%s] type[%s] has subgraph. Current not support.",
             node->GetName().c_str(), node->GetType().c_str(),
             peer_node->GetName().c_str(), peer_node->GetType().c_str());

      return GRAPH_SUCCESS;
    }
    // if peer node is PartionedCall, return owner graph's correspond node
    auto sub_graph = GetSubgraph(*peer_node, 0);
    GE_CHECK_NOTNULL(sub_graph);
    auto sub_graph_netoutput = sub_graph->FindFirstNodeMatchType(NETOUTPUT);
    GE_CHECK_NOTNULL(sub_graph_netoutput);

    for (const auto &in_data_anchor : sub_graph_netoutput->GetAllInDataAnchors()) {
      auto in_desc = sub_graph_netoutput->GetOpDesc()->MutableInputDesc(in_data_anchor->GetIdx());
      GE_CHECK_NOTNULL(in_desc);
      int32_t ref_o = 0;
      if (!AttrUtils::GetInt(in_desc, kRefIndex, ref_o)) {
        return GRAPH_FAILED;
      }
      if (peer_out_anchor_index != ref_o) {
        continue;
      }
      peer_node = NodeUtils::GetInDataNodeByIndex(*sub_graph_netoutput, in_data_anchor->GetIdx());
      GE_CHECK_NOTNULL(peer_node);
      peer_out_anchor_index = in_data_anchor->GetPeerOutAnchor()->GetIdx();
      GELOGD("in node[%s] peer_node[%s] type[%s] out anchor index[%d].", node->GetName().c_str(),
             peer_node->GetName().c_str(), peer_node->GetType().c_str(), peer_out_anchor_index);
      break;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus NodeUtils::SetNodeParallelGroup(Node &node, const char *group_name) {
  if (group_name == nullptr) {
    GE_LOGE("[Check][Parameter]Get nullptr when set parallel group on node:%s", node.GetName().c_str());
    REPORT_INNER_ERROR("E19999", "Get nullptr when set parallel group on node:%s", node.GetName().c_str());
    return GRAPH_FAILED;
  }
  std::string current_group;
  const std::string new_group(group_name);
  if (AttrUtils::GetStr(node.GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, current_group)) {
    if (new_group != current_group) {
      GE_LOGE("[Compare][Attr]Failed to set parallel group name %s on node %s, group conflict with existing %s",
              new_group.c_str(), node.GetName().c_str(), group_name);
      REPORT_INNER_ERROR("E19999", "Failed to set parallel group name %s on node %s, group conflict with existing %s",
                         new_group.c_str(), node.GetName().c_str(), group_name);
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
  if (!AttrUtils::SetStr(node.GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, new_group)) {
    GE_LOGE("[Set][Attr] Failed to set parallel group name %s on node %s",
            group_name, node.GetName().c_str());
    REPORT_INNER_ERROR("E19999", "Failed to set parallel group name %s on node %s",
                       group_name, node.GetName().c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus NodeUtils::UpdateInputOriginalShapeAndShape(const Node &node, uint32_t index, const GeShape &shape) {
  const auto desc = node.GetOpDesc();
  if (desc == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  const auto input_desc = desc->MutableInputDesc(index);
  if (input_desc == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  input_desc->SetShape(shape);
  input_desc->SetOriginShape(shape);
  return GRAPH_SUCCESS;
}

graphStatus NodeUtils::UpdateOutputOriginalShapeAndShape(const Node &node, uint32_t index, const GeShape &shape) {
  const auto desc = node.GetOpDesc();
  if (desc == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  const auto output_desc = desc->MutableOutputDesc(index);
  if (output_desc == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  output_desc->SetShape(shape);
  output_desc->SetOriginShape(shape);
  return GRAPH_SUCCESS;
}
}  // namespace ge
