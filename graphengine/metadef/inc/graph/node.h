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

#ifndef INC_GRAPH_NODE_H_
#define INC_GRAPH_NODE_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>
#include "graph/ge_attr_value.h"
#include "utils/attr_utils.h"

#include "graph/op_desc.h"
#include "graph/range_vistor.h"

namespace ge {
class ComputeGraph;

using ComputeGraphPtr = std::shared_ptr<ComputeGraph>;

class Node;

using NodePtr = std::shared_ptr<Node>;
using ConstNodePtr = std::shared_ptr<const Node>;
using NodeRef = std::weak_ptr<Node>;

class Anchor;

using AnchorPtr = std::shared_ptr<Anchor>;

class DataAnchor;

using DataAnchorPtr = std::shared_ptr<DataAnchor>;

class InDataAnchor;

using InDataAnchorPtr = std::shared_ptr<InDataAnchor>;

class OutDataAnchor;

using OutDataAnchorPtr = std::shared_ptr<OutDataAnchor>;

class ControlAnchor;

using ControlAnchorPtr = std::shared_ptr<ControlAnchor>;

class InControlAnchor;

using InControlAnchorPtr = std::shared_ptr<InControlAnchor>;

class OutControlAnchor;

using OutControlAnchorPtr = std::shared_ptr<OutControlAnchor>;

using OpDescPtr = std::shared_ptr<OpDesc>;

using ConstNode = const Node;

using NodeToOutAnchor = std::pair<NodePtr, OutDataAnchorPtr>;

using kFusionDataFlowVec_t = std::vector<std::multimap<std::string, ge::AnchorPtr>>;

// Node is a component of ComputeGraph
class Node : public std::enable_shared_from_this<Node> {
  friend class ComputeGraph;
  friend class ComputeGraphImpl;
  friend class ModelSerializeImp;

 public:
  class NodeImpl;
  using NodeImplPtr = std::shared_ptr<NodeImpl>;
  template <class T>
  using Vistor = RangeVistor<T, std::shared_ptr<ConstNode>>;
  ~Node();
  Node(const Node &) = delete;
  Node &operator=(const Node &) = delete;
  bool operator==(const Node &r_node) const;

  graphStatus Init();

  std::string GetName() const;
  std::string GetType() const;

  ComputeGraphPtr GetOwnerComputeGraph() const;
  graphStatus SetOwnerComputeGraph(const ComputeGraphPtr &graph);
  graphStatus ClearOwnerGraph(const ComputeGraphPtr &graph);

  Vistor<InDataAnchorPtr> GetAllInDataAnchors() const;
  Vistor<OutDataAnchorPtr> GetAllOutDataAnchors() const;
  uint32_t GetAllInDataAnchorsSize() const;
  uint32_t GetAllOutDataAnchorsSize() const;
  Vistor<AnchorPtr> GetAllOutAnchors() const;
  Vistor<AnchorPtr> GetAllInAnchors() const;
  InDataAnchorPtr GetInDataAnchor(const int32_t idx) const;
  OutDataAnchorPtr GetOutDataAnchor(const int32_t idx) const;
  InControlAnchorPtr GetInControlAnchor() const;
  OutControlAnchorPtr GetOutControlAnchor() const;
  Vistor<NodePtr> GetInNodes() const;
  Vistor<NodePtr> GetOutNodes() const;
  AnchorPtr GetInAnchor(const int32_t idx) const;
  AnchorPtr GetOutAnchor(const int32_t idx) const;

  bool IsAllInNodesSeen(std::unordered_set<Node *> &nodes_seen) const;

  // All in Data nodes
  Vistor<NodePtr> GetInDataNodes() const;
  // All in Control nodes
  Vistor<NodePtr> GetInControlNodes() const;
  // All in Data nodes and Control nodes
  Vistor<NodePtr> GetInAllNodes() const;

  // All out Data nodes
  Vistor<NodePtr> GetOutDataNodes() const;
  uint32_t GetOutDataNodesSize() const;
  // All out Control nodes
  Vistor<NodePtr> GetOutControlNodes() const;
  // All out Data nodes and Control nodes
  Vistor<NodePtr> GetOutAllNodes() const;

  // Get all in data nodes and its out-anchor
  Vistor<NodeToOutAnchor> GetInDataNodesAndAnchors() const;

  // Get all out data nodes and its in-anchor
  Vistor<std::pair<NodePtr, InDataAnchorPtr>> GetOutDataNodesAndAnchors() const;

  graphStatus InferShapeAndType() const;
  graphStatus Verify() const;

  graphStatus InferOriginFormat() const;

  OpDescPtr GetOpDesc() const;

  graphStatus UpdateOpDesc(const OpDescPtr &op_desc);

  graphStatus AddLinkFrom(const NodePtr &input_node);

  graphStatus AddLinkFrom(const uint32_t &index, NodePtr input_node);

  graphStatus AddLinkFrom(const std::string &name, NodePtr input_node);

  graphStatus AddLinkFromForParse(const NodePtr &input_node);

  void AddSendEventId(const uint32_t event_id);

  void AddRecvEventId(const uint32_t event_id);

  const std::vector<uint32_t> &GetSendEventIdList() const;

  const std::vector<uint32_t> &GetRecvEventIdList() const;  /*lint !e148*/

  void GetFusionInputFlowList(kFusionDataFlowVec_t &fusion_input_list);

  void GetFusionOutputFlowList(kFusionDataFlowVec_t &fusion_output_list);

  void SetFusionInputFlowList(kFusionDataFlowVec_t &fusion_input_list);

  void SetFusionOutputFlowList(kFusionDataFlowVec_t &fusion_output_list);

  bool GetHostNode() const;
  void SetHostNode(const bool is_host);

  void SetOrigNode(const NodePtr &orignode);
  NodePtr GetOrigNode();

 protected:
  Node();
  Node(const OpDescPtr &op, const ComputeGraphPtr &owner_graph);

 private:
  bool NodeMembersAreEqual(const Node &r_node) const;
  bool NodeAttrsAreEqual(const Node &r_node) const;
  bool NodeInConnectsAreEqual(const Node &r_node) const;
  bool NodeOutConnectsAreEqual(const Node &r_node) const;
  bool NodeAnchorIsEqual(const AnchorPtr &left_anchor, const AnchorPtr &right_anchor, const size_t i) const;
  NodeImplPtr impl_;
  friend class NodeUtils;
  friend class OnnxUtils;
  friend class TuningUtils;
};
}  // namespace ge

#endif  // INC_GRAPH_NODE_H_
