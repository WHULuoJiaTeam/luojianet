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

#include "graph/node.h"
#include "debug/ge_op_types.h"
#include "debug/ge_util.h"
#include "external/graph/operator_factory.h"
#include "graph/node_impl.h"
#include "graph/operator_factory_impl.h"
#include "graph/shape_refiner.h"
#include "graph/utils/ge_ir_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"


namespace ge {
Node::NodeImpl::NodeImpl(const OpDescPtr &op, const ComputeGraphPtr &owner_graph)
    : op_(op),
      owner_graph_(owner_graph),
      in_data_anchors_(),
      out_data_anchors_(),
      in_control_anchor_(nullptr),
      out_control_anchor_(nullptr),
      attrs_(),
      has_init_(false),
      host_node_(false),
      anchor_status_updated_(false) {}

Node::NodeImpl::~NodeImpl() {
  for (const auto &in_data_anchor : in_data_anchors_) {
    if (in_data_anchor != nullptr) {
      in_data_anchor->UnlinkAll();
    }
  }
  for (const auto &out_data_anchor : out_data_anchors_) {
    if (out_data_anchor != nullptr) {
      out_data_anchor->UnlinkAll();
    }
  }
  if (in_control_anchor_ != nullptr) {
    in_control_anchor_->UnlinkAll();
  }
  if (out_control_anchor_ != nullptr) {
    out_control_anchor_->UnlinkAll();
  }
}

graphStatus Node::NodeImpl::Init(const NodePtr &node) {
  if (has_init_) {
    return GRAPH_SUCCESS;
  }
  GE_CHK_BOOL_EXEC(op_ != nullptr, REPORT_INNER_ERROR("E19999", "original OpDesc is nullptr");
                   return GRAPH_FAILED, "[Check][Param] original OpDesc is nullptr");
  size_t size = op_->GetAllInputsSize();
  for (size_t i = 0UL; i < size; i++) {
    const std::shared_ptr<InDataAnchor> anchor = ComGraphMakeShared<InDataAnchor>(node, i);
    if (anchor == nullptr) {
      REPORT_CALL_ERROR("E19999", "Current in_data_anchor is null, malloc shared_ptr failed.");
      GELOGE(GRAPH_FAILED, "[Create][InDataAnchor] Current in_data_anchor is null, malloc shared_ptr failed.");
      return GRAPH_FAILED;
    }
    in_data_anchors_.push_back(anchor);
  }
  size = op_->GetOutputsSize();
  for (size_t i = 0UL; i < size; i++) {
    const std::shared_ptr<OutDataAnchor> anchor = ComGraphMakeShared<OutDataAnchor>(node, i);
    if (anchor == nullptr) {
      REPORT_CALL_ERROR("E19999", "Current out_data_anchor is null, malloc shared_ptr failed.");
      GELOGE(GRAPH_FAILED, "[Create][OutDataAnchor] Current out_data_anchor is null, malloc shared_ptr failed.");
      return GRAPH_FAILED;
    }
    out_data_anchors_.push_back(anchor);
  }
  in_control_anchor_ = ComGraphMakeShared<InControlAnchor>(node, -1);
  out_control_anchor_ = ComGraphMakeShared<OutControlAnchor>(node, -1);
  if (in_control_anchor_ == nullptr || out_control_anchor_ == nullptr) {
    REPORT_CALL_ERROR("E19999", "Current in_control_anchor or out_control_anchor is null, malloc shared_ptr failed.");
    GELOGE(GRAPH_FAILED, "[Create][ControlAnchor] Current in_control_anchor or out_control_anchor is null, "
           "malloc shared_ptr failed.");
    return GRAPH_FAILED;
  }
  has_init_ = true;
  return GRAPH_SUCCESS;
}

std::string Node::NodeImpl::GetName() const {
  GE_CHK_BOOL_EXEC(op_ != nullptr, REPORT_INNER_ERROR("E19999", "original OpDesc is nullptr");
                   return std::string(), "[Check][Param] original OpDesc is nullptr");
  return op_->GetName();
}

std::string Node::NodeImpl::GetType() const {
  GE_CHK_BOOL_EXEC(op_ != nullptr, REPORT_INNER_ERROR("E19999", "original OpDesc is nullptr");
                   return std::string(), "[Check][Param] original OpDesc is nullptr");
  return op_->GetType();
}

bool Node::NodeImpl::NodeAttrsAreEqual(const NodeImpl &r_node) const {
  const auto &attr_map = this->attrs_;
  const auto &r_attr_map = r_node.attrs_;
  // 1.Verify node's std::map<std::string, AttrValue> size
  if (attr_map.size() != r_attr_map.size()) {
    REPORT_INNER_ERROR("E19999", "param node attr map size:%zu not equal to this attr map size:%zu, "
                       "verify failed, node name: %s.", r_attr_map.size(), attr_map.size(), this->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] Size of node's attr map verify failed, node name: %s.",
           this->GetName().c_str());
    return false;
  }
  // 2.Verify node's std::map<std::string, AttrValue> key, verify values is temporarily not implemented
  for (const auto &it : attr_map) {
    if (r_attr_map.count(it.first) == 0U) {
      REPORT_INNER_ERROR("E19999", "Key of node's attr map verify failed, node name: %s key name: %s.",
                         this->GetName().c_str(), it.first.c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] Key of node's attr map verify failed, node name: %s key name: %s.",
             this->GetName().c_str(), it.first.c_str());
      return false;
    }
  }
  return true;
}

bool Node::NodeImpl::NodeMembersAreEqual(const NodeImpl &r_node) const {
  return ((((this->op_ != nullptr) && (r_node.op_ != nullptr) && (IsEqual(*(this->op_), *(r_node.op_), "node.op_"))) ||
           ((this->op_ == nullptr) && (r_node.op_ == nullptr))) &&
          IsEqual(this->has_init_, r_node.has_init_, "node.has_init_") &&
          IsEqual(this->anchor_status_updated_, r_node.anchor_status_updated_, "node.anchor_status_updated_") &&
          IsEqual(this->send_event_id_list_, r_node.send_event_id_list_, "node.send_event_id_list_") &&
          IsEqual(this->recv_event_id_list_, r_node.recv_event_id_list_, "node.recv_event_id_list_"));
}

bool Node::NodeImpl::NodeAnchorIsEqual(const AnchorPtr &left_anchor,
                                       const AnchorPtr &right_anchor,
                                       const size_t i) const {
  GE_IF_BOOL_EXEC(left_anchor == nullptr, REPORT_INNER_ERROR("E19999", "left_anchor is nullptr, check invalid.");
                  GELOGE(GRAPH_FAILED, "[Check][Param] left_anchor is null."); return false);
  GE_IF_BOOL_EXEC(right_anchor == nullptr, REPORT_INNER_ERROR("E19999", "right_anchor is nullptr, check invalid.");
                  GELOGE(GRAPH_FAILED, "[Check][Param] right_anchor is null."); return false);
  const auto anchor_peer_size = left_anchor->GetPeerAnchors().size();
  const auto right_anchor_peer_size = right_anchor->GetPeerAnchors().size();
  // Firstly, verify anchor's peer anchors size equal or not
  if (anchor_peer_size != right_anchor_peer_size) {
    REPORT_INNER_ERROR("E19999", "Size of anchor's peer anchors verify failed, node name: %s "
                       "anchor_peer_size [%zu]  is different form [%zu] at index [%zu].",
                       this->GetName().c_str(), anchor_peer_size, right_anchor_peer_size, i);
    GELOGE(GRAPH_FAILED, "[Check][Param] Size of anchor's peer anchors verify failed, node name: %s "
           "anchor_peer_size [%zu]  is different form [%zu] at index [%zu].",
           this->GetName().c_str(), anchor_peer_size, right_anchor_peer_size, i);
    return false;
  }
  // Secondly, verify anchor's peer anchor owner node equal or not
  for (size_t j = 0UL; j < anchor_peer_size; j++) {
    const auto &peer_node = left_anchor->GetPeerAnchors().at(j)->GetOwnerNode();
    const auto &r_peer_node = right_anchor->GetPeerAnchors().at(j)->GetOwnerNode();
    if (peer_node == nullptr || r_peer_node == nullptr) {
      REPORT_CALL_ERROR("E19999", "anchor's peer node is null, node name: %s index[%zu] peer node index[%zu].",
                        this->GetName().c_str(), i, j);
      GELOGE(GRAPH_FAILED, "[Get][OwnerNode] anchor's peer node is null, node name: %s index[%zu] "
             "peer node index[%zu].", this->GetName().c_str(), i, j);
      return false;
    }
    // Determine the connection relationship by linking the node's name
    if (peer_node->GetName() != r_peer_node->GetName()) {
      REPORT_INNER_ERROR("E19999", "anchor's peer node name verify failed, node name: %s index[%zu]"
                         "peer node name %s is different from %s at index [%zu].",
                         this->GetName().c_str(), i, peer_node->GetName().c_str(), r_peer_node->GetName().c_str(), j);
      GELOGE(GRAPH_FAILED, "[Check][Param] anchor's peer node name verify failed, node name: %s index[%zu]"
             "peer node name %s is different from %s at index [%zu].",
             this->GetName().c_str(), i, peer_node->GetName().c_str(), r_peer_node->GetName().c_str(), j);
      return false;
    }
  }
  return true;
}

graphStatus Node::NodeImpl::AddLinkFrom(const NodePtr &input_node, const NodePtr &owner_node) {
  // This function is deprecated, please use other two overloaded functions
  GE_CHECK_NOTNULL(input_node);
  // Input_node ---> this
  auto out_anchors = input_node->GetAllOutDataAnchors();
  if (out_anchors.size() != 1UL) {
    REPORT_INNER_ERROR("E19999", "node:%s out_anchor size is:%zu, only support 1",
                       input_node->GetName().c_str(), out_anchors.size());
    GELOGE(GRAPH_FAILED, "[Check][Param] out_anchor size is:%zu, only support 1", out_anchors.size());
    return GRAPH_PARAM_INVALID;
  }
  GE_CHK_BOOL_EXEC(op_ != nullptr, REPORT_INNER_ERROR("E19999", "original OpDesc is nullptr");
                   return GRAPH_FAILED, "[Check][Param] original OpDesc is nullptr");
  const auto op_desc = input_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  if (op_->AddInputDesc(op_desc->GetOutputDesc(0U)) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "add input desc failed, op:%s.", op_->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Add][InputDesc] failed.");
    return GRAPH_FAILED;
  }
  const std::shared_ptr<InDataAnchor> anchor = ComGraphMakeShared<InDataAnchor>(owner_node, in_data_anchors_.size());
  if (anchor == nullptr) {
    REPORT_CALL_ERROR("E19999", "out_anchor size is:%zu, malloc shared_ptr failed.", out_anchors.size());
    GELOGE(GRAPH_FAILED, "[Create][InDataAnchor] out_anchor size is:%zu, malloc shared_ptr failed.",
           out_anchors.size());
    return GRAPH_FAILED;
  }
  in_data_anchors_.push_back(anchor);
  (void) out_anchors.at(0U)->LinkTo(in_data_anchors_.back());

  return GRAPH_SUCCESS;
}

graphStatus Node::NodeImpl::AddLinkFrom(const uint32_t &index,
                                        const NodePtr &input_node,
                                        const NodePtr &owner_node) {
  GE_CHECK_NOTNULL(input_node);
  // Input_node ---> this
  auto out_anchors = input_node->GetAllOutDataAnchors();
  if (out_anchors.size() != 1UL) {
    REPORT_INNER_ERROR("E19999", "node:%s out_anchor size is:%zu, only support 1",
                       input_node->GetName().c_str(), out_anchors.size());
    GELOGE(GRAPH_FAILED, "[Check][Param] out_anchor size is:%zu, only support 1", out_anchors.size());
    return GRAPH_PARAM_INVALID;
  }

  GE_CHECK_NOTNULL(op_);
  const auto op_desc = input_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  if (op_->AddInputDesc(index, op_desc->GetOutputDesc(0U)) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "add input desc failed, index:%u.", index);
    GELOGE(GRAPH_FAILED, "[Add][InputDesc] failed.");
    return GRAPH_FAILED;
  }

  if (index < GetAllInDataAnchors(owner_node).size()) {
    (void) out_anchors.at(0U)->LinkTo(in_data_anchors_[static_cast<size_t>(index)]);
  } else {
    const std::shared_ptr<InDataAnchor>
        anchor = ComGraphMakeShared<InDataAnchor>(owner_node, in_data_anchors_.size());
    if (anchor == nullptr) {
      REPORT_CALL_ERROR("E19999", "out_anchor size is:%zu, malloc shared_ptr failed.", out_anchors.size());
      GELOGE(GRAPH_FAILED, "[Create][InDataAnchor] out_anchor size is:%zu, malloc shared_ptr failed.",
             out_anchors.size());
      return GRAPH_FAILED;
    }
    in_data_anchors_.push_back(anchor);
    (void) out_anchors.at(0U)->LinkTo(in_data_anchors_.back());
  }

  return GRAPH_SUCCESS;
}

graphStatus Node::NodeImpl::AddLinkFromForParse(const NodePtr &input_node, const NodePtr &owner_node) {
  //  This function is used for ParseWeights.
  GE_CHECK_NOTNULL(input_node);
  // Input_node ---> this
  auto out_anchors = input_node->GetAllOutDataAnchors();
  if (out_anchors.size() != 1UL) {
    REPORT_INNER_ERROR("E19999", "node:%s out_anchor size is:%zu, only support 1",
                       input_node->GetName().c_str(), out_anchors.size());
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] out_anchor size is:%zu, only support 1", out_anchors.size());
    return GRAPH_PARAM_INVALID;
  }

  const std::shared_ptr<InDataAnchor> anchor = ComGraphMakeShared<InDataAnchor>(owner_node, in_data_anchors_.size());
  if (anchor == nullptr) {
    REPORT_CALL_ERROR("E19999", "out_anchor size is:%zu, make anchor failed", out_anchors.size());
    GELOGE(GRAPH_FAILED, "[Create][InDataAnchor] out_anchor size is:%zu, make anchor failed", out_anchors.size());
    return GRAPH_FAILED;
  }
  in_data_anchors_.push_back(anchor);
  (void)out_anchors.at(0U)->LinkTo(in_data_anchors_.back());

  return GRAPH_SUCCESS;
}

graphStatus Node::NodeImpl::AddLinkFrom(const std::string &name, const NodePtr &input_node, const NodePtr &owner_node) {
  GE_CHECK_NOTNULL(input_node);
  // Input_node ---> this
  auto out_anchors = input_node->GetAllOutDataAnchors();
  if (out_anchors.size() != 1UL) {
    REPORT_INNER_ERROR("E19999", "node:%s out_anchor size is:%zu, only support 1",
                       input_node->GetName().c_str(), out_anchors.size());
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] out_anchor size is:%zu, only support 1", out_anchors.size());
    return GRAPH_PARAM_INVALID;
  }

  GE_CHECK_NOTNULL(op_);
  const auto input_op_desc = input_node->GetOpDesc();
  GE_CHECK_NOTNULL(input_op_desc);
  const auto index = op_->GetInputIndexByName(name);
  if (index != -1) {
    if (index >= static_cast<int32_t>(in_data_anchors_.size())) {
      REPORT_INNER_ERROR("E19999", "op %s get input name %s 's index %d is illegal as which >= indataanchors size:%zu.",
                         op_->GetName().c_str(), name.c_str(), index, in_data_anchors_.size());
      GELOGE(GRAPH_FAILED, "[Check][Param] op %s get input name %s 's index %d is illegal.",
             op_->GetName().c_str(), name.c_str(), index);
      return GRAPH_FAILED;
    }
    (void) out_anchors.at(0U)->LinkTo(in_data_anchors_[static_cast<size_t>(index)]);
  } else {
    const std::shared_ptr<InDataAnchor>
        anchor = ComGraphMakeShared<InDataAnchor>(owner_node, in_data_anchors_.size());
    GE_CHECK_NOTNULL(anchor);
    in_data_anchors_.push_back(anchor);
    (void) out_anchors.at(0U)->LinkTo(in_data_anchors_.back());
  }
  if (op_->AddInputDesc(name, input_op_desc->GetOutputDesc(0U)) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "add input desc failed, name:%s, op:%s", name.c_str(), op_->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Add][InputDesc] failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

ComputeGraphPtr Node::NodeImpl::GetOwnerComputeGraph() const {
  return owner_graph_.lock();
}

graphStatus Node::NodeImpl::SetOwnerComputeGraph(const ComputeGraphPtr &graph) {
  if (graph == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  owner_graph_ = graph;
  return GRAPH_SUCCESS;
}

graphStatus Node::NodeImpl::ClearOwnerGraph(const ComputeGraphPtr &graph) {
  owner_graph_ = graph;
  return GRAPH_SUCCESS;
}

Node::Vistor<InDataAnchorPtr> Node::NodeImpl::GetAllInDataAnchors(const ConstNodePtr &node_ptr) const {
  return Node::Vistor<InDataAnchorPtr>(node_ptr, in_data_anchors_);
}

Node::Vistor<OutDataAnchorPtr> Node::NodeImpl::GetAllOutDataAnchors(const ConstNodePtr &node_ptr) const {
  return Node::Vistor<OutDataAnchorPtr>(node_ptr, out_data_anchors_);
}

uint32_t Node::NodeImpl::GetAllInDataAnchorsSize() const {
  return static_cast<uint32_t>(in_data_anchors_.size());
}

uint32_t Node::NodeImpl::GetAllOutDataAnchorsSize() const {
  return static_cast<uint32_t>(out_data_anchors_.size());
}

Node::Vistor<AnchorPtr> Node::NodeImpl::GetAllInAnchors(const ConstNodePtr &owner_node) const {
  std::vector<AnchorPtr> vec;
  // Push back in_data_anchors_
  for (const auto &in_anchor_iter : Node::Vistor<InDataAnchorPtr>(owner_node, in_data_anchors_)) {
    const auto in_anchor = Anchor::DynamicAnchorCast<Anchor>(in_anchor_iter);
    if (in_anchor != nullptr) {
      vec.push_back(in_anchor);
    }
  }
  // Push back in_control_anchor_
  if ((in_control_anchor_->GetPeerOutControlAnchors().size() > 0UL) ||
      (in_control_anchor_->GetPeerOutDataAnchors().size() > 0UL)) {
    const auto in_anchor = Anchor::DynamicAnchorCast<Anchor>(in_control_anchor_);
    if (in_anchor != nullptr) {
      vec.push_back(in_anchor);
    }
  }
  return Node::Vistor<AnchorPtr>(owner_node, vec);
}

Node::Vistor<AnchorPtr> Node::NodeImpl::GetAllOutAnchors(const ConstNodePtr &owner_node) const {
  std::vector<AnchorPtr> vec;
  // Push back out_data_anchors_
  for (const auto &out_anchor_iter : Node::Vistor<OutDataAnchorPtr>(owner_node, out_data_anchors_)) {
    const auto out_anchor = Anchor::DynamicAnchorCast<Anchor>(out_anchor_iter);
    if (out_anchor != nullptr) {
      vec.push_back(out_anchor);
    }
  }
  // Push back out_control_anchor_
  if ((out_control_anchor_->GetPeerInControlAnchors().size() > 0UL) ||
      (out_control_anchor_->GetPeerInDataAnchors().size() > 0UL)) {
    const auto out_anchor = Anchor::DynamicAnchorCast<Anchor>(out_control_anchor_);
    if (out_anchor != nullptr) {
      vec.push_back(out_anchor);
    }
  }
  return Node::Vistor<AnchorPtr>(owner_node, vec);
}

InDataAnchorPtr Node::NodeImpl::GetInDataAnchor(const int32_t idx) const {
  if ((idx < 0) || (idx >= static_cast<int32_t>(in_data_anchors_.size()))) {
    GELOGW("[Check][Param] Op %s doesn't have data input %d, type = %s", GetName().c_str(), idx, GetType().c_str());
    return nullptr;
  } else {
    return in_data_anchors_[static_cast<size_t>(idx)];
  }
}

AnchorPtr Node::NodeImpl::GetInAnchor(const int32_t idx) const {
  // Idx can't be less than -1 or >= in_data_anchors_.size(), -1 means index of control anchor_
  if ((idx < -1) || (idx >= static_cast<int32_t>(in_data_anchors_.size()))) {
    GELOGW("[Check][Param] Op %s doesn't have input %d, type = %s", GetName().c_str(), idx, GetType().c_str());
    return nullptr;
  } else {
    // Return control anchor
    if (idx == -1) {
      return Anchor::DynamicAnchorCast<Anchor>(in_control_anchor_);
    }
    // Return data anchor
    return in_data_anchors_[static_cast<size_t>(idx)];
  }
}

AnchorPtr Node::NodeImpl::GetOutAnchor(const int32_t idx) const {
  // Idx can't be less than -1 or >= out_data_anchors_.size(), -1 means index of control anchor_
  if ((idx < -1) || (idx >= static_cast<int32_t>(out_data_anchors_.size()))) {
    REPORT_INNER_ERROR("E19999", "Op:%s(%s) doesn't have index:%d's anchorname",
                       GetName().c_str(), GetType().c_str(), idx);
    GELOGE(GRAPH_FAILED, "[Check][Param] Op[%s] doesn't have index[%d]'s out_anchor which optype is %s.",
           GetName().c_str(), idx, GetType().c_str());
    return nullptr;
  } else {
    // Return control anchor
    if (idx == -1) {
      return Anchor::DynamicAnchorCast<Anchor>(out_control_anchor_);
    }
    // Return data anchor
    return out_data_anchors_[static_cast<size_t>(idx)];
  }
}

OutDataAnchorPtr Node::NodeImpl::GetOutDataAnchor(const int32_t idx) const {
  if ((idx < 0) || (idx >= static_cast<int32_t>(out_data_anchors_.size()))) {
    REPORT_INNER_ERROR("E19999", "Op:%s(%s) doesn't have index:%d's anchorname",
                       GetName().c_str(), GetType().c_str(), idx);
    GELOGE(GRAPH_FAILED, "[Check][Param] Op[%s] doesn't have index[%d]'s out_data_anchor which optype is %s.",
           GetName().c_str(), idx, GetType().c_str());
    return nullptr;
  } else {
    return out_data_anchors_[static_cast<size_t>(idx)];
  }
}

InControlAnchorPtr Node::NodeImpl::GetInControlAnchor() const {
  return in_control_anchor_;
}

OutControlAnchorPtr Node::NodeImpl::GetOutControlAnchor() const {
  return out_control_anchor_;
}

Node::Vistor<NodePtr> Node::NodeImpl::GetInNodes(const ge::ConstNodePtr &owner_node) const {
  std::vector<NodePtr> vec;
  for (const auto &in_anchor : in_data_anchors_) {
    GE_CHK_BOOL_EXEC((in_anchor != nullptr),
                     continue, "[Check][Param] node:%s in_data_anchor is nullptr", GetName().c_str());
    const auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
      continue;
    }
    const auto node = out_anchor->GetOwnerNode();
    vec.push_back(node);
  }
  if (in_control_anchor_ != nullptr) {
    if (in_control_anchor_->IsPeerOutAnchorsEmpty()) {
      return Node::Vistor<NodePtr>(owner_node, vec);
    }

    const auto peer_out_anchors = in_control_anchor_->GetPeerOutDataAnchors();
    for (const auto &out_anchor : peer_out_anchors) {
      GE_CHK_BOOL_EXEC(out_anchor != nullptr, continue,
                       "[Check][Param] node:%s in_control_anchor_ peer out data anchors is nullptr",
                       GetName().c_str());
      const auto node = out_anchor->GetOwnerNode();
      vec.push_back(node);
    }

    auto peer_out_control_anchors = in_control_anchor_->GetPeerOutControlAnchors();
    for (const auto &out_control_anchor : peer_out_control_anchors) {
      const auto node = out_control_anchor->GetOwnerNode();
      vec.push_back(node);
    }
  }
  return Node::Vistor<NodePtr>(owner_node, vec);
}

bool Node::NodeImpl::IsAllInNodesSeen(std::unordered_set<Node *> &nodes_seen) const {
  for (const auto &in_anchor : in_data_anchors_) {
    GE_CHK_BOOL_EXEC((in_anchor != nullptr),
                     continue, "[Check][Param] in_data_anchor is nullptr, node:%s", GetName().c_str());
    const auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
      continue;
    }
    const auto node = out_anchor->GetOwnerNode();
    if ((node->GetType() == NEXTITERATION) || (node->GetType() == REFNEXTITERATION)) {
      continue;
    }
    if (nodes_seen.count(node.get()) == 0U) {
      return false;
    }
  }

  if (in_control_anchor_ != nullptr) {
    if (in_control_anchor_->IsPeerOutAnchorsEmpty()) {
      return true;
    }
    const auto peer_out_control_anchors = in_control_anchor_->GetPeerOutControlAnchors();
    for (const auto &out_control_anchor : peer_out_control_anchors) {
      const auto node = out_control_anchor->GetOwnerNode();
      if ((node->GetType() == NEXTITERATION) || (node->GetType() == REFNEXTITERATION)) {
        continue;
      }
      if (nodes_seen.count(node.get()) == 0U) {
        return false;
      }
    }
  }

  return true;
}

Node::Vistor<NodePtr> Node::NodeImpl::GetInDataNodes(const ge::ConstNodePtr &owner_node) const {
  std::vector<NodePtr> vec;
  for (const auto &in_anchor : in_data_anchors_) {
    GE_CHK_BOOL_EXEC((in_anchor != nullptr), continue,
                     "[Check][Param] in_data_anchor is nullptr, node:%s", GetName().c_str());
    auto anchor_ptr = in_anchor->GetPeerOutAnchor();
    if (anchor_ptr == nullptr) {
      continue;
    }
    const auto node = anchor_ptr->GetOwnerNode();
    vec.push_back(node);
  }
  return Node::Vistor<NodePtr>(owner_node, vec);
}

Node::Vistor<NodePtr> Node::NodeImpl::GetInControlNodes(const ge::ConstNodePtr &owner_node) const {
  std::vector<NodePtr> vec;
  if (in_control_anchor_ != nullptr) {
    for (const auto &in_anchor : in_control_anchor_->GetPeerOutControlAnchors()) {
      const auto node = in_anchor->GetOwnerNode();
      vec.push_back(node);
    }
  }
  return Node::Vistor<NodePtr>(owner_node, vec);
}

Node::Vistor<NodePtr> Node::NodeImpl::GetOutNodes(const ge::ConstNodePtr &owner_node) const {
  std::vector<NodePtr> vec;
  for (const auto &out_anchor : out_data_anchors_) {
    GE_CHK_BOOL_EXEC((out_anchor != nullptr), continue,
                     "[Check][Param] out data anchor is nullptr, node:%s", GetName().c_str());
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      const auto node = peer_in_anchor->GetOwnerNode();
      vec.push_back(node);
    }
  }
  if (out_control_anchor_ != nullptr) {
    auto peer_in_control_anchors = out_control_anchor_->GetPeerInControlAnchors();
    for (const auto &in_control_anchor : peer_in_control_anchors) {
      const auto node = in_control_anchor->GetOwnerNode();
      vec.push_back(node);
    }
  }
  return Node::Vistor<NodePtr>(owner_node, vec);
}

Node::Vistor<NodePtr> Node::NodeImpl::GetInAllNodes(const ge::ConstNodePtr &owner_node) const {
  std::vector<NodePtr> vec;
  for (const auto &in_node : GetInDataNodes(owner_node)) {
    vec.push_back(in_node);
  }
  for (const auto &in_control_node : GetInControlNodes(owner_node)) {
    vec.push_back(in_control_node);
  }
  return Node::Vistor<NodePtr>(owner_node, vec);
}

Node::Vistor<NodePtr> Node::NodeImpl::GetOutDataNodes(const std::shared_ptr<const Node> &owner_node) const {
  std::vector<NodePtr> vec;
  for (const auto &out_anchor : out_data_anchors_) {
    GE_CHK_BOOL_EXEC((out_anchor != nullptr), continue,
                     "[Check][Param] out data anchor is nullptr, node:%s", GetName().c_str());
    for (const auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHK_BOOL_EXEC((in_anchor != nullptr), continue,
                       "[Check][Param] out anchor GetPeerInDataAnchors is nullptr, node:%s", GetName().c_str());
      const auto node = in_anchor->GetOwnerNode();
      vec.push_back(node);
    }
  }
  return Node::Vistor<NodePtr>(owner_node, vec);
}

uint32_t Node::NodeImpl::GetOutDataNodesSize() const {
  uint32_t out_nums = 0U;
  for (const auto &out_anchor : out_data_anchors_) {
    GE_CHK_BOOL_EXEC((out_anchor != nullptr), continue,
                     "[Check][Param] out data anchor is nullptr, node:%s", GetName().c_str());
    out_nums += out_anchor->GetPeerInDataNodesSize();
  }
  return out_nums;
}

Node::Vistor<NodePtr> Node::NodeImpl::GetOutControlNodes(const ge::ConstNodePtr &owner_node) const {
  std::vector<NodePtr> vec;

  for (const auto &out_anchor : out_data_anchors_) {
    GE_CHK_BOOL_EXEC((out_anchor != nullptr), continue,
                     "[Check][Param] out data anchor is nullptr, node:%s", GetName().c_str());
    for (const auto &in_anchor : out_anchor->GetPeerInControlAnchors()) {
      GE_CHK_BOOL_EXEC((in_anchor != nullptr), continue,
                       "[Check][Param] Peer In Control Anchor is nullptr, node:%s", GetName().c_str());
      const auto in_node = in_anchor->GetOwnerNode();
      vec.push_back(in_node);
    }
  }

  if (out_control_anchor_ != nullptr) {
    for (const auto &in_anchor : out_control_anchor_->GetPeerAnchors()) {
      const auto in_node = in_anchor->GetOwnerNode();
      vec.push_back(in_node);
    }
  }

  return Node::Vistor<NodePtr>(owner_node, vec);
}

Node::Vistor<NodePtr> Node::NodeImpl::GetOutAllNodes(const ge::ConstNodePtr &owner_node) const {
  std::vector<NodePtr> vec;
  for (const auto &out_anchor : out_data_anchors_) {
    GE_CHK_BOOL_EXEC((out_anchor != nullptr), { continue; },
                     "[Check][Param] out data anchor is nullptr, node:%s", GetName().c_str());
    for (const auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
      const auto node = in_anchor->GetOwnerNode();
      vec.push_back(node);
    }
    for (const auto &in_anchor : out_anchor->GetPeerInControlAnchors()) {
      GE_CHK_BOOL_EXEC(in_anchor != nullptr, continue,
                       "[Check][Param] node:%s Peer In Control Anchor is nullptr", GetName().c_str());
      const auto node = in_anchor->GetOwnerNode();
      vec.push_back(node);
    }
  }

  if (out_control_anchor_ != nullptr) {
    for (const auto &in_anchor : out_control_anchor_->GetPeerAnchors()) {
      const auto node = in_anchor->GetOwnerNode();
      vec.push_back(node);
    }
  }
  return Node::Vistor<NodePtr>(owner_node, vec);
}

graphStatus Node::NodeImpl::InferShapeAndType(const ge::ConstNodePtr &owner_node) const {
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(owner_node);
  return ShapeRefiner::InferShapeAndType(owner_node, op);
}

graphStatus Node::NodeImpl::InferOriginFormat(const ge::ConstNodePtr &owner_node) const {
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(owner_node);
  // Get infer func and execute
  GE_CHK_BOOL_EXEC(op_ != nullptr, REPORT_INNER_ERROR("E19999", "original OpDesc is nullptr, check invalid.");
                   return GRAPH_FAILED, "[Check][Param] original OpDesc is nullptr");
  return op_->CallInferFormatFunc(op);
}

graphStatus Node::NodeImpl::Verify(const ge::ConstNodePtr &owner_node) const {
  const std::string data_type = "Data";
  const std::string aipp_data_type = "AippData";
  const std::string const_type = "Const";
  const std::string const_type_train = "Constant";
  const std::string variable_type = "Variable";
  const bool is_unknown_graph = GetOwnerComputeGraph()->GetGraphUnknownFlag();
  GE_CHK_BOOL_EXEC(op_ != nullptr, REPORT_INNER_ERROR("E19999", "original OpDesc is nullptr, check invalid.");
                   return GRAPH_FAILED, "[Check][Param] original OpDesc is nullptr");

  if (!is_unknown_graph) {
    for (const auto &in_anchor_ptr : GetAllInDataAnchors(owner_node)) {
      if (in_anchor_ptr == nullptr) {
        GELOGW("[Verify][CheckParam] In data anchor is null");
        continue;
      }
      const bool valid_anchor = (op_->GetType() == data_type) || (op_->GetType() == aipp_data_type) ||
                                (op_->GetType() == const_type) || (op_->GetType() == variable_type) ||
                                (op_->GetType() == const_type_train) ||
                                (op_->MutableInputDesc(static_cast<uint32_t>(in_anchor_ptr->GetIdx())) == nullptr) ||
                                (in_anchor_ptr->GetPeerAnchors().size() > 0UL);
      if (!valid_anchor) {
        ErrorManager::GetInstance().ATCReportErrMessage("E11019", {"opname", "index"},
                                                        {GetName(), std::to_string(in_anchor_ptr->GetIdx())});
        GELOGE(GRAPH_FAILED, "[Check][Param] operator %s's input %d is not linked.",
               GetName().c_str(), in_anchor_ptr->GetIdx());
        return GRAPH_FAILED;
      }
    }
  }

  const std::string frameworkop_type = "FrameworkOp";
  const bool need_update_name = (op_->GetType() != frameworkop_type) && (!is_unknown_graph);
  if (need_update_name) {
    const auto node_op = ge::OperatorFactoryImpl::CreateOperator("node_op", op_->GetType());
    if (node_op.IsEmpty()) {
      GELOGW("[Verify][CheckParam] Get op from OperatorFactory failed, type: %s", op_->GetType().c_str());
    } else {
      GELOGD("get op from OperatorFactory success. opType: %s", op_->GetType().c_str());
      const auto temp_op_desc = ge::OpDescUtils::GetOpDescFromOperator(node_op);
      if (temp_op_desc == nullptr) {
        REPORT_INNER_ERROR("E19999", "GetOpDescFromOperator failed, as return nullptr, type:%s",
                           op_->GetType().c_str());
        GELOGE(GRAPH_FAILED, "[Get][OpDesc] temp op desc is null, type:%s", op_->GetType().c_str());
        return GRAPH_FAILED;
      }
      if (!op_->UpdateInputName(temp_op_desc->GetAllInputName())) {
        GELOGW("[Verify][Update] Update input name failed");
      }
      if (!op_->UpdateOutputName(temp_op_desc->GetAllOutputName())) {
        GELOGW("[Verify][Update] Update output name failed");
      }
    }
    node_op.BreakConnect();
  }
  GE_IF_BOOL_EXEC(is_unknown_graph, return GRAPH_SUCCESS;);
  if (op_->CommonVerify() == GRAPH_SUCCESS) {
    Operator op_proxy = ge::OpDescUtils::CreateOperatorFromNode(owner_node);
    auto verify_func = op_->GetVerifyFunc();
    if (verify_func == nullptr) {
      verify_func = OperatorFactoryImpl::GetVerifyFunc(GetType());
    }
    if (verify_func != nullptr) {
      return (graphStatus)verify_func(op_proxy);
    }
    return GRAPH_SUCCESS;
  } else {
    REPORT_CALL_ERROR("E19999", "%s(%s) Verify failed.", op_->GetName().c_str(), op_->GetType().c_str());
    GELOGE(GRAPH_FAILED, "[Call][CommonVerify] %s(%s) Verify failed.", op_->GetName().c_str(), op_->GetType().c_str());
    return GRAPH_FAILED;
  }
}

OpDescPtr Node::NodeImpl::GetOpDesc() const { return op_; }

graphStatus Node::NodeImpl::UpdateOpDesc(const OpDescPtr &op_desc) {
  GE_CHK_BOOL_EXEC(op_ != nullptr, REPORT_INNER_ERROR("E19999", "original OpDesc is nullptr");
          return GRAPH_FAILED, "[Check][Param] original OpDesc is nullptr");
  GE_CHK_BOOL_EXEC(op_desc != nullptr, REPORT_INNER_ERROR("E19999", "param op_desc is nullptr, check invalid.");
          return GRAPH_PARAM_INVALID, "[Check][Param] Param OpDesc is nullptr");
  GE_CHK_BOOL_EXEC(op_->GetInputsSize() == op_desc->GetInputsSize(),
                   REPORT_INNER_ERROR("E19999", "inputs count(%zu) of param op_desc not equal to "
                                                "inputs count(%zu) of original opdesc:%s, check invalid",
                                      op_desc->GetInputsSize(), op_->GetInputsSize(), op_->GetName().c_str());
                           return GRAPH_PARAM_INVALID,
                   "[Check][Param] Inputs count expected to be same, original OpDesc %zu, Param OpDesc %zu",
                   op_->GetInputsSize(), op_desc->GetInputsSize());

  GE_CHK_BOOL_EXEC(op_->GetOutputsSize() == op_desc->GetOutputsSize(),
                   REPORT_INNER_ERROR("E19999", "outputs count(%zu) of param op_desc not equal to "
                                                "outputs count(%zu) of original opdesc:%s, check invalid",
                                      op_desc->GetOutputsSize(), op_->GetOutputsSize(), op_->GetName().c_str());
                           return GRAPH_PARAM_INVALID,
                   "[Check][Param] Outputs count expected to be same, original OpDesc %zu, Param OpDesc %zu",
                   op_->GetOutputsSize(), op_desc->GetOutputsSize());
  op_ = op_desc;
  return GRAPH_SUCCESS;
}

Node::Vistor<std::pair<NodePtr, OutDataAnchorPtr>> Node::NodeImpl::GetInDataNodesAndAnchors(
    const ConstNodePtr &owner_node) const {
  std::vector<std::pair<NodePtr, OutDataAnchorPtr>> vec;
  for (const auto &p : in_data_anchors_) {
    if (p == nullptr) {
      GELOGW("[Check][Param] In data anchor is nullptr, node=%s, type=%s", GetType().c_str(), GetName().c_str());
      continue;
    }
    auto anchor_ptr = p->GetPeerOutAnchor();
    if (anchor_ptr == nullptr) {
      continue;
    }
    auto node = anchor_ptr->GetOwnerNode();
    if (node == nullptr) {
      GELOGW("[Check][Param] Src node is nullptr, node=%s, type=%s", GetType().c_str(), GetName().c_str());
      continue;
    }
    vec.push_back(std::make_pair(node, anchor_ptr));
  }
  return Node::Vistor<std::pair<NodePtr, OutDataAnchorPtr>>(owner_node, vec);
}

Node::Vistor<std::pair<NodePtr, InDataAnchorPtr>> Node::NodeImpl::GetOutDataNodesAndAnchors(
    const ConstNodePtr &owner_node) const {
  std::vector<std::pair<NodePtr, InDataAnchorPtr>> vec;
  for (const auto &p : out_data_anchors_) {
    if (p == nullptr) {
      GELOGW("[Check][Param] Out data anchor is nullptr, node=%s, type=%s", GetType().c_str(), GetName().c_str());
      continue;
    }
    for (const auto &in_anchor : p->GetPeerInDataAnchors()) {
      if (in_anchor == nullptr) {
        GELOGW("[Check][Param] Dst in data anchor is nullptr, node=%s, type=%s", GetType().c_str(), GetName().c_str());
        continue;
      }
      auto node = in_anchor->GetOwnerNode();
      if (node == nullptr) {
        GELOGW("[Check][Param] Dst node is nullptr, node=%s, type=%s", GetType().c_str(), GetName().c_str());
        continue;
      }
      vec.push_back(std::make_pair(node, in_anchor));
    }
  }
  return Node::Vistor<std::pair<NodePtr, InDataAnchorPtr>>(owner_node, vec);
}

Node::Node()
    : enable_shared_from_this(), impl_(std::shared_ptr<NodeImpl>(new NodeImpl())) {}

Node::Node(const OpDescPtr &op, const ComputeGraphPtr &owner_graph)
    : enable_shared_from_this(), impl_(std::shared_ptr<NodeImpl>(new NodeImpl(op, owner_graph))) {}

Node::~Node() {}

graphStatus Node::Init() {
  return impl_->Init(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::string Node::GetName() const {
  return impl_->GetName();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::string Node::GetType() const {
  return impl_->GetType();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool Node::NodeAttrsAreEqual(const Node &r_node) const {
  return impl_->NodeAttrsAreEqual(*(r_node.impl_));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool Node::NodeMembersAreEqual(const Node &r_node) const {
  return impl_->NodeMembersAreEqual(*(r_node.impl_));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool Node::NodeAnchorIsEqual(const AnchorPtr &left_anchor,
                                                                            const AnchorPtr &right_anchor,
                                                                            const size_t i) const {
  return impl_->NodeAnchorIsEqual(left_anchor, right_anchor, i);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool Node::NodeInConnectsAreEqual(const Node &r_node) const {
  // 1.Verify all in data and control anchors size
  const auto in_data_anchor_size = this->GetAllInDataAnchors().size();
  const auto r_in_data_anchor_size = r_node.GetAllInDataAnchors().size();
  if (in_data_anchor_size != r_in_data_anchor_size) {
    REPORT_INNER_ERROR("E19999", "param node in data anchors count:%zu not equal to "
                       "this in data anchors count:%zu, verify failed, node name: %s.",
                       r_in_data_anchor_size, in_data_anchor_size, this->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] Size of node's in data anchors verify failed, node name: %s.",
           this->GetName().c_str());
    return false;
  }
  const auto l_in_anchors = this->GetAllInAnchors();
  const auto r_in_anchors = r_node.GetAllInAnchors();
  // Data anchors size equal, all anchors size not equal, means control anchor size not equal
  const auto in_control_anchor_size = l_in_anchors.size() - in_data_anchor_size;
  const auto r_in_control_anchor_size = r_in_anchors.size() - r_in_data_anchor_size;
  if (in_control_anchor_size != r_in_control_anchor_size) {
    REPORT_INNER_ERROR("E19999", "param node in control anchors count:%zu not equal to "
                       "this in control anchors count:%zu, verify failed, node name: %s.",
                       r_in_control_anchor_size, in_control_anchor_size, this->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] Size of node's in control anchors verify failed, node name: %s.",
           this->GetName().c_str());
    return false;
  }
  // 2.Verify all in data and control anchors connect info
  for (size_t i = 0UL; i < this->GetAllInAnchors().size(); i++) {
    // Verify data anchors
    if (i < in_data_anchor_size) {
      const auto &in_anchor = l_in_anchors.at(i);
      const auto &r_in_anchor = r_in_anchors.at(i);
      if (!(NodeAnchorIsEqual(in_anchor, r_in_anchor, i))) {
        GELOGE(GRAPH_FAILED, "[Call][NodeAnchorIsEqual] Node's in data control anchor verify failed, node name: %s.",
               this->GetName().c_str());
        return false;
      }
    } else {
      // Verify control anchors
      const auto &in_control_anchor = l_in_anchors.at(i);
      const auto &r_in_control_anchor = r_in_anchors.at(i);
      if (!(NodeAnchorIsEqual(in_control_anchor, r_in_control_anchor, i - in_data_anchor_size))) {
        GELOGE(GRAPH_FAILED, "[Call][NodeAnchorIsEqual] Node's in control anchor verify failed, node name: %s.",
               this->GetName().c_str());
        return false;
      }
    }
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool Node::NodeOutConnectsAreEqual(const Node &r_node) const {
  // 1.Verify all out data and control anchors size
  const auto l_out_data_anchors = this->GetAllOutDataAnchors();
  const auto r_out_data_anchors = r_node.GetAllOutDataAnchors();
  const auto out_data_anchor_size = l_out_data_anchors.size();
  const auto r_out_data_anchor_size = r_out_data_anchors.size();
  if (out_data_anchor_size != r_out_data_anchor_size) {
    REPORT_INNER_ERROR("E19999", "param node out data anchors count:%zu not equal to "
                       "this out data anchors count:%zu, verify failed, node name: %s.",
                       r_out_data_anchor_size, out_data_anchor_size, this->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] Size of node's out data anchors verify failed, node name: %s.",
           this->GetName().c_str());
    return false;
  }
  const auto l_out_anchors = this->GetAllOutAnchors();
  const auto r_out_anchors = r_node.GetAllOutAnchors();
  // Data anchors size equal, all anchors size not equal, means control anchor size not equal
  const auto out_control_anchor_size = l_out_anchors.size() - out_data_anchor_size;
  const auto r_out_control_anchor_size = r_out_anchors.size() - r_out_data_anchor_size;
  if (out_control_anchor_size != r_out_control_anchor_size) {
    REPORT_INNER_ERROR("E19999", "param node out control anchors count:%zu not equal to "
                       "this out control anchors count:%zu, verify failed, node name: %s.",
                       r_out_control_anchor_size, out_control_anchor_size, this->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] Size of node's out control anchors verify failed, node name: %s.",
           this->GetName().c_str());
    return false;
  }

  // 2.Verify all out data and control anchors connect info
  for (size_t i = 0UL; i < this->GetAllOutAnchors().size(); i++) {
    // Verify data anchors
    if (i < out_data_anchor_size) {
      const auto &out_anchor = l_out_data_anchors.at(i);
      const auto &r_out_anchor = r_out_data_anchors.at(i);
      if (!(NodeAnchorIsEqual(out_anchor, r_out_anchor, i))) {
        GELOGE(GRAPH_FAILED, "[Call][NodeAnchorIsEqual] Node's out data control anchor verify failed, node name: %s.",
               this->GetName().c_str());
        return false;
      }
    } else {
      // Verify control anchors
      const auto &out_control_anchor = l_out_anchors.at(i);
      const auto &r_out_control_anchor = r_out_anchors.at(i);
      if (!(NodeAnchorIsEqual(out_control_anchor, r_out_control_anchor, i - out_data_anchor_size))) {
        GELOGE(GRAPH_FAILED, "[Call][NodeAnchorIsEqual] Node's out control anchor verify failed, node name: %s.",
               this->GetName().c_str());
        return false;
      }
    }
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool Node::operator==(const Node &r_node) const {
  return (NodeMembersAreEqual(r_node) && NodeAttrsAreEqual(r_node) && NodeInConnectsAreEqual(r_node) &&
          NodeOutConnectsAreEqual(r_node));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus Node::AddLinkFrom(const NodePtr &input_node) {
  return impl_->AddLinkFrom(input_node, shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus Node::AddLinkFrom(const uint32_t &index,
                                                                             NodePtr input_node) {
  return impl_->AddLinkFrom(index, input_node, shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus Node::AddLinkFromForParse(const NodePtr &input_node) {
  return impl_->AddLinkFromForParse(input_node, shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus Node::AddLinkFrom(const std::string &name, NodePtr input_node) {
  return impl_->AddLinkFrom(name, input_node, shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraphPtr Node::GetOwnerComputeGraph() const {
  return impl_->GetOwnerComputeGraph();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus Node::SetOwnerComputeGraph(const ComputeGraphPtr &graph) {
  return impl_->SetOwnerComputeGraph(graph);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus Node::ClearOwnerGraph(const ComputeGraphPtr &graph) {
  return impl_->ClearOwnerGraph(graph);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<InDataAnchorPtr> Node::GetAllInDataAnchors() const {
  return impl_->GetAllInDataAnchors(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<OutDataAnchorPtr> Node::GetAllOutDataAnchors() const {
  return impl_->GetAllOutDataAnchors(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t Node::GetAllInDataAnchorsSize() const {
  return impl_->GetAllInDataAnchorsSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t Node::GetAllOutDataAnchorsSize() const {
  return impl_->GetAllOutDataAnchorsSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<AnchorPtr> Node::GetAllInAnchors() const {
  return impl_->GetAllInAnchors(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<AnchorPtr> Node::GetAllOutAnchors() const {
  return impl_->GetAllOutAnchors(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InDataAnchorPtr Node::GetInDataAnchor(const int32_t idx) const {
  return impl_->GetInDataAnchor(idx);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY AnchorPtr Node::GetInAnchor(const int32_t idx) const {
  return impl_->GetInAnchor(idx);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY AnchorPtr Node::GetOutAnchor(const int32_t idx) const {
  return impl_->GetOutAnchor(idx);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OutDataAnchorPtr Node::GetOutDataAnchor(const int32_t idx) const {
  return impl_->GetOutDataAnchor(idx);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InControlAnchorPtr Node::GetInControlAnchor() const {
  return impl_->GetInControlAnchor();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OutControlAnchorPtr Node::GetOutControlAnchor() const {
  return impl_->GetOutControlAnchor();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetInNodes() const {
  return impl_->GetInNodes(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool Node::IsAllInNodesSeen(
    std::unordered_set<Node *> &nodes_seen) const {
  return impl_->IsAllInNodesSeen(nodes_seen);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetInDataNodes() const {
  return impl_->GetInDataNodes(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetInControlNodes() const {
  return impl_->GetInControlNodes(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetOutNodes() const {
  return impl_->GetOutNodes(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetInAllNodes() const {
  return impl_->GetInAllNodes(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetOutDataNodes() const {
  return impl_->GetOutDataNodes(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t Node::GetOutDataNodesSize() const {
  return impl_->GetOutDataNodesSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetOutControlNodes() const {
  return impl_->GetOutControlNodes(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetOutAllNodes() const {
  return impl_->GetOutAllNodes(shared_from_this());
}

graphStatus Node::InferShapeAndType() const {
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(shared_from_this());
  return ShapeRefiner::InferShapeAndType(shared_from_this(), op);
}

graphStatus Node::InferOriginFormat() const {
  return impl_->InferOriginFormat(shared_from_this());
}

graphStatus Node::Verify() const {
  return impl_->Verify(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr Node::GetOpDesc() const {
  return impl_->GetOpDesc();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus Node::UpdateOpDesc(const OpDescPtr &op_desc) {
  return impl_->UpdateOpDesc(op_desc);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<std::pair<NodePtr, OutDataAnchorPtr>>
Node::GetInDataNodesAndAnchors() const {
  return impl_->GetInDataNodesAndAnchors(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<std::pair<NodePtr, InDataAnchorPtr>>
Node::GetOutDataNodesAndAnchors() const {
  return impl_->GetOutDataNodesAndAnchors(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void Node::AddSendEventId(const uint32_t event_id) {
  impl_->AddSendEventId(event_id);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void Node::AddRecvEventId(const uint32_t event_id) {
  impl_->AddRecvEventId(event_id);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::vector<uint32_t> &Node::GetSendEventIdList() const {
  return impl_->GetSendEventIdList();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::vector<uint32_t> &Node::GetRecvEventIdList() const {
  return impl_->GetRecvEventIdList();
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY  void Node::GetFusionInputFlowList(
    kFusionDataFlowVec_t &fusion_input_list) {
  impl_->GetFusionInputFlowList(fusion_input_list);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void Node::GetFusionOutputFlowList(
    kFusionDataFlowVec_t &fusion_output_list) {
  impl_->GetFusionOutputFlowList(fusion_output_list);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void Node::SetFusionInputFlowList(
    kFusionDataFlowVec_t &fusion_input_list) {
  impl_->SetFusionInputFlowList(fusion_input_list);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void Node::SetFusionOutputFlowList(
    kFusionDataFlowVec_t &fusion_output_list) {
  impl_->SetFusionOutputFlowList(fusion_output_list);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool Node::GetHostNode() const {
  return impl_->GetHostNode();
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void Node::SetHostNode(const bool is_host) {
  impl_->SetHostNode(is_host);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void Node::SetOrigNode(const NodePtr &orignode) {
  impl_->SetOrigNode(orignode);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY NodePtr Node::GetOrigNode() {
  return impl_->GetOrigNode();
}
}  // namespace ge
