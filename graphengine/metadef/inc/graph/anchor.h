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

#ifndef INC_GRAPH_ANCHOR_H_
#define INC_GRAPH_ANCHOR_H_

#include "graph/compiler_options.h"

#include <memory>
#include "graph/ge_error_codes.h"
#include "graph/range_vistor.h"
#include "graph/types.h"
#include "graph/node.h"

namespace ge {
enum AnchorStatus {
  ANCHOR_SUSPEND = 0,  // dat null
  ANCHOR_CONST = 1,
  ANCHOR_DATA = 2,  // Effective
  ANCHOR_RESERVED = 3
};
using ConstAnchor = const Anchor;
class AnchorImpl;
using AnchorImplPtr = std::shared_ptr<AnchorImpl>;

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Anchor : public std::enable_shared_from_this<Anchor> {
  friend class AnchorUtils;

 public:
  using TYPE = const char *;
  template <class T>
  using Vistor = RangeVistor<T, std::shared_ptr<ConstAnchor>>;

  Anchor(const NodePtr& owner_node, const int32_t idx);
  Anchor(const Anchor &) = delete;
  Anchor(Anchor &&) = delete;
  Anchor &operator=(const Anchor &) = delete;
  Anchor &operator=(Anchor &&) = delete;
  virtual ~Anchor();

 protected:
  // Whether the two anchor is equal
  virtual bool Equal(const AnchorPtr anchor) const = 0;
  virtual bool IsTypeOf(const TYPE type) const;

 public:
  // Get all peer anchors connected to current anchor
  Vistor<AnchorPtr> GetPeerAnchors() const;
  // Get peer anchor size
  size_t GetPeerAnchorsSize() const;
  // Get first peer anchor
  AnchorPtr GetFirstPeerAnchor() const;

  // Get the anchor belong to which node
  NodePtr GetOwnerNode() const;

  // Remove all links with the anchor
  void UnlinkAll() noexcept;

  // Remove link with the given anchor
  graphStatus Unlink(const AnchorPtr &peer);

  // Replace peer with new peers
  graphStatus ReplacePeer(const AnchorPtr &old_peer, const AnchorPtr &first_peer, const AnchorPtr &second_peer);

  // Judge if the anchor is linked with the given anchor
  bool IsLinkedWith(const AnchorPtr &peer);

  // Get anchor index of the node
  int32_t GetIdx() const;

  // set anchor index of the node
  void SetIdx(const int32_t index);

 protected:
  template <class T>
  static Anchor::TYPE TypeOf() {
    static_assert(std::is_base_of<Anchor, T>::value, "T must be a Anchor!");
    return METADEF_FUNCTION_IDENTIFIER;
  }

 public:
  template <class T>
  static std::shared_ptr<T> DynamicAnchorCast(const AnchorPtr anchorPtr) {
    static_assert(std::is_base_of<Anchor, T>::value, "T must be a Anchor!");
    if ((anchorPtr == nullptr) || (!anchorPtr->IsTypeOf<T>())) {
      return nullptr;
    }
    return std::static_pointer_cast<T>(anchorPtr);
  }

  template <typename T>
  bool IsTypeOf() const {
    return IsTypeOf(TypeOf<T>());
  }

 protected:
  AnchorImplPtr impl_;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY DataAnchor : public Anchor {
  friend class AnchorUtils;

 public:
  explicit DataAnchor(const NodePtr &owner_node, const int32_t idx);
  DataAnchor(const DataAnchor &) = delete;
  DataAnchor &operator=(const DataAnchor &) = delete;
  DataAnchor(DataAnchor &&) = delete;
  DataAnchor &operator=(DataAnchor &&) = delete;
  virtual ~DataAnchor() = default;

 protected:
  bool IsTypeOf(const TYPE type) const override;

 private:
  Format format_{FORMAT_ND};
  AnchorStatus status_{ANCHOR_SUSPEND};
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InDataAnchor : public DataAnchor {
  friend class OutDataAnchor;

  friend class OutControlAnchor;

 public:
  explicit InDataAnchor(const NodePtr &owner_node, const int32_t idx);

  virtual ~InDataAnchor() = default;

  // Get  source out data anchor
  OutDataAnchorPtr GetPeerOutAnchor() const;

  // Build connection from OutDataAnchor to InDataAnchor
  graphStatus LinkFrom(const OutDataAnchorPtr &src);

 protected:
  bool Equal(const AnchorPtr anchor) const override;
  bool IsTypeOf(const TYPE type) const override;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OutDataAnchor : public DataAnchor {
  friend class InDataAnchor;

  friend class AnchorUtils;

 public:
  template <class T>
  using Vistor = RangeVistor<T, std::shared_ptr<ConstAnchor>>;

  explicit OutDataAnchor(const NodePtr &owner_node, const int32_t idx);

  virtual ~OutDataAnchor() = default;
  // Get dst in data anchor(one or more)
  Vistor<InDataAnchorPtr> GetPeerInDataAnchors() const;
  uint32_t GetPeerInDataNodesSize() const;

  // Get dst in control anchor(one or more)
  Vistor<InControlAnchorPtr> GetPeerInControlAnchors() const;

  // Build connection from OutDataAnchor to InDataAnchor
  graphStatus LinkTo(const InDataAnchorPtr &dest);

  // Build connection from OutDataAnchor to InControlAnchor
  graphStatus LinkTo(const InControlAnchorPtr &dest);

 protected:
  bool Equal(const AnchorPtr anchor) const override;
  bool IsTypeOf(const TYPE type) const override;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ControlAnchor : public Anchor {
 public:
  explicit ControlAnchor(const NodePtr &owner_node);

  explicit ControlAnchor(const NodePtr &owner_node, const int32_t idx);
  virtual ~ControlAnchor() = default;

 protected:
  bool IsTypeOf(const TYPE type) const override;
  ControlAnchor(const ControlAnchor &) = delete;
  ControlAnchor &operator=(const ControlAnchor &) = delete;
  ControlAnchor(ControlAnchor &&) = delete;
  ControlAnchor &operator=(ControlAnchor &&) = delete;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InControlAnchor : public ControlAnchor {
  friend class OutControlAnchor;

  friend class OutDataAnchor;

 public:
  explicit InControlAnchor(const NodePtr &owner_node);

  explicit InControlAnchor(const NodePtr &owner_node, const int32_t idx);

  virtual ~InControlAnchor() = default;

  // Get  source out control anchors
  Vistor<OutControlAnchorPtr> GetPeerOutControlAnchors() const;
  bool IsPeerOutAnchorsEmpty() const;

  // Get  source out data anchors
  Vistor<OutDataAnchorPtr> GetPeerOutDataAnchors() const;

  // Build connection from OutControlAnchor to InControlAnchor
  graphStatus LinkFrom(const OutControlAnchorPtr &src);

 protected:
  bool Equal(const AnchorPtr anchor) const override;
  bool IsTypeOf(const TYPE type) const override;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OutControlAnchor : public ControlAnchor {
  friend class InControlAnchor;

 public:
  template <class T>
  using Vistor = RangeVistor<T, std::shared_ptr<ConstAnchor>>;

  explicit OutControlAnchor(const NodePtr &owner_node);

  explicit OutControlAnchor(const NodePtr &owner_node, const int32_t idx);

  virtual ~OutControlAnchor() = default;

  // Get dst in control anchor(one or more)
  Vistor<InControlAnchorPtr> GetPeerInControlAnchors() const;
  // Get dst data anchor in control anchor(one or more)
  Vistor<InDataAnchorPtr> GetPeerInDataAnchors() const;

  // Build connection from OutControlAnchor to InControlAnchor
  graphStatus LinkTo(const InControlAnchorPtr &dest);
  // Build connection from OutDataAnchor to InDataAnchor
  graphStatus LinkTo(const InDataAnchorPtr &dest);

 protected:
  bool Equal(const AnchorPtr anchor) const override;
  bool IsTypeOf(const TYPE type) const override;
};
}  // namespace ge
#endif  // INC_GRAPH_ANCHOR_H_
