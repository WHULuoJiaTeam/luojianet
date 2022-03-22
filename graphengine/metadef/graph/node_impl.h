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

#ifndef GRAPH_NODE_IMPL_H_
#define GRAPH_NODE_IMPL_H_

#include "graph/node.h"

namespace ge {
class Node::NodeImpl {
 public:
  NodeImpl() = default;
  NodeImpl(const OpDescPtr &op, const ComputeGraphPtr &owner_graph);
  ~NodeImpl();
  graphStatus Init(const NodePtr &node);
  std::string GetName() const;
  std::string GetType() const;
  bool NodeAttrsAreEqual(const NodeImpl &r_node) const;
  bool NodeMembersAreEqual(const NodeImpl &r_node) const;
  bool NodeAnchorIsEqual(const AnchorPtr &left_anchor,
                         const AnchorPtr &right_anchor,
                         const size_t i) const;

  graphStatus AddLinkFrom(const NodePtr &input_node, const NodePtr &owner_node);
  graphStatus AddLinkFrom(const uint32_t &index,
                          const NodePtr &input_node,
                          const NodePtr &owner_node);
  graphStatus AddLinkFromForParse(const NodePtr &input_node, const NodePtr &owner_node);
  graphStatus AddLinkFrom(const std::string &name, const NodePtr &input_node, const NodePtr &owner_node);

  ComputeGraphPtr GetOwnerComputeGraph() const;
  graphStatus SetOwnerComputeGraph(const ComputeGraphPtr &graph);
  graphStatus ClearOwnerGraph(const ComputeGraphPtr &graph);

  Node::Vistor<InDataAnchorPtr> GetAllInDataAnchors(const ConstNodePtr &node_ptr) const;
  Node::Vistor<OutDataAnchorPtr> GetAllOutDataAnchors(const ConstNodePtr &node_ptr) const;
  uint32_t GetAllInDataAnchorsSize() const;
  uint32_t GetAllOutDataAnchorsSize() const;
  Node::Vistor<AnchorPtr> GetAllInAnchors(const ConstNodePtr &owner_node) const;
  Node::Vistor<AnchorPtr> GetAllOutAnchors(const ConstNodePtr &owner_node) const;
  InDataAnchorPtr GetInDataAnchor(const int32_t idx) const;
  AnchorPtr GetInAnchor(const int32_t idx) const;
  AnchorPtr GetOutAnchor(const int32_t idx) const;
  OutDataAnchorPtr GetOutDataAnchor(const int32_t idx) const;
  InControlAnchorPtr GetInControlAnchor() const;
  OutControlAnchorPtr GetOutControlAnchor() const;

  Node::Vistor<NodePtr> GetInNodes(const ConstNodePtr &owner_node) const;
  bool IsAllInNodesSeen(std::unordered_set<Node *> &nodes_seen) const;
  Node::Vistor<NodePtr> GetInDataNodes(const ConstNodePtr &owner_node) const;
  Node::Vistor<NodePtr> GetInControlNodes(const ConstNodePtr &owner_node) const;
  Node::Vistor<NodePtr> GetOutNodes(const ConstNodePtr &owner_node) const;
  Node::Vistor<NodePtr> GetInAllNodes(const ConstNodePtr &owner_node) const;
  Node::Vistor<NodePtr> GetOutDataNodes(const ConstNodePtr &owner_node) const;
  uint32_t GetOutDataNodesSize() const;
  Node::Vistor<NodePtr> GetOutControlNodes(const ConstNodePtr &owner_node) const;
  Node::Vistor<NodePtr> GetOutAllNodes(const ConstNodePtr &owner_node) const;

  graphStatus InferShapeAndType(const ConstNodePtr &owner_node) const;
  graphStatus InferOriginFormat(const ConstNodePtr &owner_node) const;
  graphStatus Verify(const ConstNodePtr &owner_node) const;
  OpDescPtr GetOpDesc() const;
  graphStatus UpdateOpDesc(const OpDescPtr &op_desc);
  Node::Vistor<std::pair<NodePtr, OutDataAnchorPtr>>
  GetInDataNodesAndAnchors(const ConstNodePtr &owner_node) const;
  Node::Vistor<std::pair<NodePtr, InDataAnchorPtr>>
  GetOutDataNodesAndAnchors(const ConstNodePtr &owner_node) const;

  void AddSendEventId(uint32_t event_id) { send_event_id_list_.push_back(event_id); }
  void AddRecvEventId(uint32_t event_id) { recv_event_id_list_.push_back(event_id); }

  const std::vector<uint32_t> &GetSendEventIdList() const { return send_event_id_list_; }
  const std::vector<uint32_t> &GetRecvEventIdList() const { return recv_event_id_list_; }

  void GetFusionInputFlowList(kFusionDataFlowVec_t &fusion_input_list) const {
    fusion_input_list = fusion_input_dataflow_list_;
  }
  void GetFusionOutputFlowList(kFusionDataFlowVec_t &fusion_output_list) const {
    fusion_output_list = fusion_output_dataflow_list_;
  }
  void SetFusionInputFlowList(const kFusionDataFlowVec_t &fusion_input_list) {
    fusion_input_dataflow_list_ = fusion_input_list;
  }
  void SetFusionOutputFlowList(const kFusionDataFlowVec_t &fusion_output_list) {
    fusion_output_dataflow_list_ = fusion_output_list;
  }

  bool GetHostNode() const { return host_node_; }
  void SetHostNode(bool is_host) { host_node_ = is_host; }

  void SetOrigNode(const NodePtr &orignode) { orig_node_ = orignode; }
  NodePtr GetOrigNode() { return orig_node_; }

 private:
  friend class NodeUtils;
  friend class TuningUtils;
  friend class OnnxUtils;
  OpDescPtr op_;
  std::weak_ptr<ComputeGraph> owner_graph_;
  std::vector<InDataAnchorPtr> in_data_anchors_;
  std::vector<OutDataAnchorPtr> out_data_anchors_;
  InControlAnchorPtr in_control_anchor_;
  OutControlAnchorPtr out_control_anchor_;
  std::map<std::string, GeAttrValue> attrs_;
  bool has_init_{false};
  bool host_node_{false};
  bool anchor_status_updated_{false};
  std::vector<uint32_t> send_event_id_list_;
  std::vector<uint32_t> recv_event_id_list_;

  kFusionDataFlowVec_t fusion_input_dataflow_list_;
  kFusionDataFlowVec_t fusion_output_dataflow_list_;
  NodePtr orig_node_{nullptr};
};
}  // namespace ge
#endif  // GRAPH_BUFFER_IMPL_H_
