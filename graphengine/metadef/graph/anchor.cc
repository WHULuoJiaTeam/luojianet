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

#include "graph/anchor.h"
#include <algorithm>
#include <cstring>
#include "debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/node.h"

namespace ge {
class AnchorImpl {
 public:
  AnchorImpl(const NodePtr &owner_node, int32_t idx);
  ~AnchorImpl() = default;
  size_t GetPeerAnchorsSize() const;
  Anchor::Vistor<AnchorPtr> GetPeerAnchors(const std::shared_ptr<ConstAnchor> &anchor_ptr) const;
  AnchorPtr GetFirstPeerAnchor() const;
  NodePtr GetOwnerNode() const;
  int32_t GetIdx() const;
  void SetIdx(int32_t index);

  // All peer anchors connected to current anchor
  std::vector<std::weak_ptr<Anchor>> peer_anchors_;
  // The owner node of anchor
  std::weak_ptr<Node> owner_node_;
  // The index of current anchor
  int32_t idx_;
};

AnchorImpl::AnchorImpl(const NodePtr &owner_node, int32_t idx) : owner_node_(owner_node), idx_(idx) {}

size_t AnchorImpl::GetPeerAnchorsSize() const {
  return peer_anchors_.size();
}

Anchor::Vistor<AnchorPtr> AnchorImpl::GetPeerAnchors(
    const std::shared_ptr<ConstAnchor> &anchor_ptr) const {
  std::vector<AnchorPtr> ret;
  for (const auto &anchor : peer_anchors_) {
    ret.push_back(anchor.lock());
  }
  return Anchor::Vistor<AnchorPtr>(anchor_ptr, ret);
}

AnchorPtr AnchorImpl::GetFirstPeerAnchor() const {
  if (peer_anchors_.empty()) {
    return nullptr;
  } else {
    return Anchor::DynamicAnchorCast<Anchor>(peer_anchors_.begin()->lock());
  }
}

NodePtr AnchorImpl::GetOwnerNode() const { return owner_node_.lock(); }

int32_t AnchorImpl::GetIdx() const { return idx_; }

void AnchorImpl::SetIdx(int32_t index) { idx_ = index; }

Anchor::Anchor(const NodePtr &owner_node, const int32_t idx)
    : enable_shared_from_this(), impl_(std::shared_ptr<AnchorImpl>(new AnchorImpl(owner_node, idx))) {}

Anchor::~Anchor() = default;

bool Anchor::IsTypeOf(const TYPE type) const { return std::string(Anchor::TypeOf<Anchor>()) == std::string(type); }

size_t Anchor::GetPeerAnchorsSize() const {
  return impl_->GetPeerAnchorsSize();
}

Anchor::Vistor<AnchorPtr> Anchor::GetPeerAnchors() const {
  return impl_->GetPeerAnchors(shared_from_this());
}

AnchorPtr Anchor::GetFirstPeerAnchor() const {
  return impl_->GetFirstPeerAnchor();
}

NodePtr Anchor::GetOwnerNode() const {
  return impl_->GetOwnerNode();
}

void Anchor::UnlinkAll() noexcept {
  if (!impl_->peer_anchors_.empty()) {
    do {
      const auto peer_anchor_ptr = impl_->peer_anchors_.begin()->lock();
      (void)Unlink(peer_anchor_ptr);
    } while (!impl_->peer_anchors_.empty());
  }
}

graphStatus Anchor::Unlink(const AnchorPtr &peer) {
  if (peer == nullptr) {
    REPORT_INNER_ERROR("E19999", "param peer is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] peer anchor is invalid.");
    return GRAPH_FAILED;
  }
  const auto it = std::find_if(impl_->peer_anchors_.begin(), impl_->peer_anchors_.end(),
      [peer](const std::weak_ptr<Anchor> &an) {
    auto anchor = an.lock();
    return peer->Equal(anchor);
  });

  if (it == impl_->peer_anchors_.end()) {
    GELOGW("[Check][Param] Unlink failed , as this anchor is not connected to peer.");
    return GRAPH_FAILED;
  }

  const auto it_peer = std::find_if(peer->impl_->peer_anchors_.begin(), peer->impl_->peer_anchors_.end(),
      [this](const std::weak_ptr<Anchor> &an) {
        const auto anchor = an.lock();
        return Equal(anchor);
      });

  GE_CHK_BOOL_RET_STATUS(it_peer != peer->impl_->peer_anchors_.end(), GRAPH_FAILED,
                         "[Check][Param] peer(%s, %d) is not connected to this anchor(%s, %d)",
                         peer->GetOwnerNode()->GetName().c_str(), peer->GetIdx(),
                         this->GetOwnerNode()->GetName().c_str(), this->GetIdx());
  (void)impl_->peer_anchors_.erase(it);
  (void)peer->impl_->peer_anchors_.erase(it_peer);
  return GRAPH_SUCCESS;
}

graphStatus Anchor::ReplacePeer(const AnchorPtr &old_peer, const AnchorPtr &first_peer, const AnchorPtr &second_peer) {
  GE_CHK_BOOL_RET_STATUS(old_peer != nullptr, GRAPH_FAILED, "[Check][Param] this old peer anchor is nullptr");
  GE_CHK_BOOL_RET_STATUS(first_peer != nullptr, GRAPH_FAILED, "[Check][Param] this first peer anchor is nullptr");
  GE_CHK_BOOL_RET_STATUS(second_peer != nullptr, GRAPH_FAILED, "[Check][Param] this second peer anchor is nullptr");
  const auto this_it = std::find_if(impl_->peer_anchors_.begin(), impl_->peer_anchors_.end(),
      [old_peer](const std::weak_ptr<Anchor> &an) {
    const auto anchor = an.lock();
    return old_peer->Equal(anchor);
  });

  GE_CHK_BOOL_RET_STATUS(this_it != impl_->peer_anchors_.end(), GRAPH_FAILED,
                         "[Check][Param] this anchor(%s, %d) is not connected to old_peer(%s, %d)",
                         this->GetOwnerNode()->GetName().c_str(), this->GetIdx(),
                         old_peer->GetOwnerNode()->GetName().c_str(), old_peer->GetIdx());

  const auto old_it = std::find_if(old_peer->impl_->peer_anchors_.begin(), old_peer->impl_->peer_anchors_.end(),
                                   [this](const std::weak_ptr<Anchor> &an) {
                                     auto anchor = an.lock();
                                     return Equal(anchor);
                                   });
  GE_CHK_BOOL_RET_STATUS(old_it != old_peer->impl_->peer_anchors_.end(), GRAPH_FAILED,
                         "[Check][Param] old_peer(%s, %d) is not connected to this anchor(%s, %d)",
                         old_peer->GetOwnerNode()->GetName().c_str(), old_peer->GetIdx(),
                         this->GetOwnerNode()->GetName().c_str(), this->GetIdx());
  *this_it = first_peer;
  first_peer->impl_->peer_anchors_.push_back(shared_from_this());
  *old_it = second_peer;
  second_peer->impl_->peer_anchors_.push_back(old_peer);
  return GRAPH_SUCCESS;
}

bool Anchor::IsLinkedWith(const AnchorPtr &peer) {
  const auto it = std::find_if(impl_->peer_anchors_.begin(), impl_->peer_anchors_.end(),
      [peer](const std::weak_ptr<Anchor> &an) {
    const auto anchor = an.lock();
    if (peer == nullptr) {
      GELOGE(GRAPH_FAILED, "[Check][Param] this old peer anchor is nullptr");
      return false;
    }
    return peer->Equal(anchor);
  });
  return (it != impl_->peer_anchors_.end());
}

int32_t Anchor::GetIdx() const {
  return impl_->GetIdx();
}

void Anchor::SetIdx(const int32_t index) {
  impl_->SetIdx(index);
}

DataAnchor::DataAnchor(const NodePtr &owner_node, const int32_t idx) : Anchor(owner_node, idx) {}

bool DataAnchor::IsTypeOf(const TYPE type) const {
  if (std::string(Anchor::TypeOf<DataAnchor>()) == std::string(type)) {
    return true;
  }
  return Anchor::IsTypeOf(type);
}

InDataAnchor::InDataAnchor(const NodePtr &owner_node, const int32_t idx) : DataAnchor(owner_node, idx) {}

OutDataAnchorPtr InDataAnchor::GetPeerOutAnchor() const {
  if (impl_ == nullptr || impl_->peer_anchors_.empty()) {
    return nullptr;
  } else {
    return Anchor::DynamicAnchorCast<OutDataAnchor>(impl_->peer_anchors_.begin()->lock());
  }
}

graphStatus InDataAnchor::LinkFrom(const OutDataAnchorPtr &src) {
  // InDataAnchor must be only linkfrom once
  if ((src == nullptr) || (src->impl_ == nullptr) ||
      (impl_ == nullptr) || !impl_->peer_anchors_.empty())  {
    REPORT_INNER_ERROR("E19999", "src anchor is invalid or the peerAnchors is not empty.");
    GELOGE(GRAPH_FAILED, "[Check][Param] src anchor is invalid or the peerAnchors is not empty.");
    return GRAPH_FAILED;
  }
  impl_->peer_anchors_.push_back(src);
  src->impl_->peer_anchors_.push_back(shared_from_this());
  return GRAPH_SUCCESS;
}

bool InDataAnchor::Equal(const AnchorPtr anchor) const {
  const auto in_data_anchor = Anchor::DynamicAnchorCast<InDataAnchor>(anchor);
  if (in_data_anchor != nullptr) {
    if ((GetOwnerNode() == in_data_anchor->GetOwnerNode()) && (GetIdx() == in_data_anchor->GetIdx())) {
      return true;
    }
  }
  return false;
}

bool InDataAnchor::IsTypeOf(const TYPE type) const {
  if (std::string(Anchor::TypeOf<InDataAnchor>()) == std::string(type)) {
    return true;
  }
  return DataAnchor::IsTypeOf(type);
}

OutDataAnchor::OutDataAnchor(const NodePtr &owner_node, const int32_t idx) : DataAnchor(owner_node, idx) {}

OutDataAnchor::Vistor<InDataAnchorPtr> OutDataAnchor::GetPeerInDataAnchors() const {
  std::vector<InDataAnchorPtr> ret;
  if (impl_ != nullptr) {
    for (const auto &anchor : impl_->peer_anchors_) {
      const auto in_data_anchor = Anchor::DynamicAnchorCast<InDataAnchor>(anchor.lock());
      if (in_data_anchor != nullptr) {
        ret.push_back(in_data_anchor);
      }
    }
  }
  return OutDataAnchor::Vistor<InDataAnchorPtr>(shared_from_this(), ret);
}

uint32_t OutDataAnchor::GetPeerInDataNodesSize() const {
  uint32_t out_nums = 0U;
  if (impl_ != nullptr) {
    for (const auto &anchor : impl_->peer_anchors_) {
      const auto in_data_anchor = Anchor::DynamicAnchorCast<InDataAnchor>(anchor.lock());
      if (in_data_anchor != nullptr && in_data_anchor->GetOwnerNode() != nullptr) {
        out_nums++;
      }
    }
  }
  return out_nums;
}

OutDataAnchor::Vistor<InControlAnchorPtr> OutDataAnchor::GetPeerInControlAnchors() const {
  std::vector<InControlAnchorPtr> ret;
  if (impl_ != nullptr) {
    for (const auto &anchor : impl_->peer_anchors_) {
      const auto in_control_anchor = Anchor::DynamicAnchorCast<InControlAnchor>(anchor.lock());
      if (in_control_anchor != nullptr) {
        ret.push_back(in_control_anchor);
      }
    }
  }
  return OutDataAnchor::Vistor<InControlAnchorPtr>(shared_from_this(), ret);
}

graphStatus OutDataAnchor::LinkTo(const InDataAnchorPtr &dest) {
  if ((dest == nullptr) || (dest->impl_ == nullptr) ||
      !dest->impl_->peer_anchors_.empty()) {
    REPORT_INNER_ERROR("E19999", "dest anchor is nullptr or the peerAnchors is not empty.");
    GELOGE(GRAPH_FAILED, "[Check][Param] dest anchor is nullptr or the peerAnchors is not empty.");
    return GRAPH_FAILED;
  }
  if (impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "anchor param is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] owner anchor is invalid.");;
    return GRAPH_FAILED;
  }
  impl_->peer_anchors_.push_back(dest);
  dest->impl_->peer_anchors_.push_back(shared_from_this());
  return GRAPH_SUCCESS;
}

graphStatus OutDataAnchor::LinkTo(const InControlAnchorPtr &dest) {
  if (dest == nullptr || dest->impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "param dest is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] dest anchor is invalid.");
    return GRAPH_FAILED;
  }
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "src anchor is invalid.");
    return GRAPH_FAILED;
  }
  impl_->peer_anchors_.push_back(dest);
  dest->impl_->peer_anchors_.push_back(shared_from_this());
  return GRAPH_SUCCESS;
}

graphStatus OutControlAnchor::LinkTo(const InDataAnchorPtr &dest) {
  if (dest == nullptr || dest->impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "param dest is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] dest anchor is invalid.");
    return GRAPH_FAILED;
  }
  if (impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "anchor param is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] owner anchor is invalid.");;
    return GRAPH_FAILED;
  }
  impl_->peer_anchors_.push_back(dest);
  dest->impl_->peer_anchors_.push_back(shared_from_this());
  return GRAPH_SUCCESS;
}

bool OutDataAnchor::Equal(const AnchorPtr anchor) const {
  CHECK_FALSE_EXEC(anchor != nullptr, return false);
  const auto out_data_anchor = Anchor::DynamicAnchorCast<OutDataAnchor>(anchor);
  if (out_data_anchor != nullptr) {
    if ((GetOwnerNode() == out_data_anchor->GetOwnerNode()) && (GetIdx() == out_data_anchor->GetIdx())) {
      return true;
    }
  }
  return false;
}

bool OutDataAnchor::IsTypeOf(const TYPE type) const {
  if (std::string(Anchor::TypeOf<OutDataAnchor>()) == std::string(type)) {
    return true;
  }
  return DataAnchor::IsTypeOf(type);
}

ControlAnchor::ControlAnchor(const NodePtr &owner_node) : Anchor(owner_node, -1) {}

ControlAnchor::ControlAnchor(const NodePtr &owner_node, const int32_t idx) : Anchor(owner_node, idx) {}

bool ControlAnchor::IsTypeOf(const TYPE type) const {
  if (std::string(Anchor::TypeOf<ControlAnchor>()) == std::string(type)) {
    return true;
  }
  return Anchor::IsTypeOf(type);
}

InControlAnchor::InControlAnchor(const NodePtr &owner_node) : ControlAnchor(owner_node) {}

InControlAnchor::InControlAnchor(const NodePtr &owner_node, const int32_t idx) : ControlAnchor(owner_node, idx) {}

InControlAnchor::Vistor<OutControlAnchorPtr> InControlAnchor::GetPeerOutControlAnchors() const {
  std::vector<OutControlAnchorPtr> ret;
  if (impl_ != nullptr) {
    for (const auto &anchor : impl_->peer_anchors_) {
      const auto out_control_anchor = Anchor::DynamicAnchorCast<OutControlAnchor>(anchor.lock());
      if (out_control_anchor != nullptr) {
        ret.push_back(out_control_anchor);
      }
    }
  }
  return InControlAnchor::Vistor<OutControlAnchorPtr>(shared_from_this(), ret);
}

bool InControlAnchor::IsPeerOutAnchorsEmpty() const {
  if (impl_ == nullptr) {
    return false;
  }
  return impl_->peer_anchors_.empty();
}

InControlAnchor::Vistor<OutDataAnchorPtr> InControlAnchor::GetPeerOutDataAnchors() const {
  std::vector<OutDataAnchorPtr> ret;
  if (impl_ != nullptr) {
    for (const auto &anchor : impl_->peer_anchors_) {
      const auto out_data_anchor = Anchor::DynamicAnchorCast<OutDataAnchor>(anchor.lock());
      if (out_data_anchor != nullptr) {
        ret.push_back(out_data_anchor);
      }
    }
  }
  return InControlAnchor::Vistor<OutDataAnchorPtr>(shared_from_this(), ret);
}

graphStatus InControlAnchor::LinkFrom(const OutControlAnchorPtr &src) {
  if (src == nullptr || src->impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "param src is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] src anchor is invalid.");
    return GRAPH_FAILED;
  }
  if (impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "anchor param is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] owner anchor is invalid.");;
    return GRAPH_FAILED;
  }
  impl_->peer_anchors_.push_back(src);
  src->impl_->peer_anchors_.push_back(shared_from_this());
  return GRAPH_SUCCESS;
}

bool InControlAnchor::Equal(const AnchorPtr anchor) const {
  CHECK_FALSE_EXEC(anchor != nullptr, REPORT_INNER_ERROR("E19999", "param anchor is nullptr, check invalid");
                   GELOGE(GRAPH_FAILED, "[Check][Param] anchor is invalid."); return false);
  const auto in_control_anchor = Anchor::DynamicAnchorCast<InControlAnchor>(anchor);
  if (in_control_anchor != nullptr) {
    if (GetOwnerNode() == in_control_anchor->GetOwnerNode()) {
      return true;
    }
  }
  return false;
}

bool InControlAnchor::IsTypeOf(const TYPE type) const {
  if (std::string(Anchor::TypeOf<InControlAnchor>()) == std::string(type)) {
    return true;
  }
  return ControlAnchor::IsTypeOf(type);
}

OutControlAnchor::OutControlAnchor(const NodePtr &owner_node) : ControlAnchor(owner_node) {}

OutControlAnchor::OutControlAnchor(const NodePtr &owner_node, const int32_t idx) : ControlAnchor(owner_node, idx) {}

OutControlAnchor::Vistor<InControlAnchorPtr> OutControlAnchor::GetPeerInControlAnchors() const {
  std::vector<InControlAnchorPtr> ret;
  if (impl_ != nullptr) {
    for (const auto &anchor : impl_->peer_anchors_) {
      const auto in_control_anchor = Anchor::DynamicAnchorCast<InControlAnchor>(anchor.lock());
      if (in_control_anchor != nullptr) {
        ret.push_back(in_control_anchor);
      }
    }
  }
  return OutControlAnchor::Vistor<InControlAnchorPtr>(shared_from_this(), ret);
}

OutControlAnchor::Vistor<InDataAnchorPtr> OutControlAnchor::GetPeerInDataAnchors() const {
  std::vector<InDataAnchorPtr> ret;
  if (impl_ != nullptr) {
    for (const auto &anchor : impl_->peer_anchors_) {
      const auto in_data_anchor = Anchor::DynamicAnchorCast<InDataAnchor>(anchor.lock());
      if (in_data_anchor != nullptr) {
        ret.push_back(in_data_anchor);
      }
    }
  }
  return OutControlAnchor::Vistor<InDataAnchorPtr>(shared_from_this(), ret);
}

graphStatus OutControlAnchor::LinkTo(const InControlAnchorPtr &dest) {
  if (dest == nullptr || dest->impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "param dest is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] dest anchor is invalid.");
    return GRAPH_FAILED;
  }
  if (impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "anchor param is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] owner anchor is invalid.");;
    return GRAPH_FAILED;
  }
  impl_->peer_anchors_.push_back(dest);
  dest->impl_->peer_anchors_.push_back(shared_from_this());
  return GRAPH_SUCCESS;
}

bool OutControlAnchor::Equal(const AnchorPtr anchor) const {
  const auto out_control_anchor = Anchor::DynamicAnchorCast<OutControlAnchor>(anchor);
  if (out_control_anchor != nullptr) {
    if (GetOwnerNode() == out_control_anchor->GetOwnerNode()) {
      return true;
    }
  }
  return false;
}

bool OutControlAnchor::IsTypeOf(const TYPE type) const {
  if (std::string(Anchor::TypeOf<OutControlAnchor>()) == std::string(type)) {
    return true;
  }
  return ControlAnchor::IsTypeOf(type);
}
}  // namespace ge
