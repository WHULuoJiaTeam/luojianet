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
#ifndef GE_GRAPH_PASSES_TRANSOP_WITHOUT_RESHAPE_FUSION_PASS_H_
#define GE_GRAPH_PASSES_TRANSOP_WITHOUT_RESHAPE_FUSION_PASS_H_

#include <vector>
#include <utility>
#include "inc/graph_pass.h"

namespace ge {
///
/// Transform operators depth fusion
///
class TransOpWithoutReshapeFusionPass : public GraphPass {
 public:
  TransOpWithoutReshapeFusionPass() {}
  virtual ~TransOpWithoutReshapeFusionPass() {}

  graphStatus Run(ge::ComputeGraphPtr graph) override;

 private:
  void SetRemainNode(const vector<pair<OutDataAnchorPtr, InDataAnchorPtr>> &nodes_anchor);
  bool FormatContinuousCheck(const OutDataAnchorPtr &out_anchor, const InDataAnchorPtr &in_anchor);
  void RemoveNousedNodes(const ComputeGraphPtr &graph);
  void GetBeginOutDescAndEndInDesc(const int index, GeTensorDesc &out_desc, GeTensorDesc &in_desc);

  void GetFormatTransferDesc(const GeTensorDesc &out_desc,
                             const GeTensorDesc &in_desc,
                             GeTensorDesc &format_transfer_input,
                             GeTensorDesc &format_transfer_output);

  void GetCastOpDesc(const GeTensorDesc &out_desc,
                     const GeTensorDesc &in_desc,
                     GeTensorDesc &cast_input,
                     GeTensorDesc &cast_output);

  graphStatus FormatFusion(const int index,
                           OpDescPtr &format_transfer_op,
                           int32_t &fusion_op_count,
                           bool &fusion_continue);

  graphStatus DataTypeFusion(const int index, OpDescPtr &cast_op, int32_t &fusion_op_count);

  void GetOutDataPeerInControlAnchors(const size_t index,
                                      vector<vector<InControlAnchorPtr>> &out_data_peer_in_control_anchors);

  void GetInControlPeerOutControlAnchors(
      const size_t index,
      vector<vector<OutControlAnchorPtr>> &in_control_peer_out_control_anchors);

  void GetOutControlPeerAnchors(
      const size_t index,
      vector<vector<InControlAnchorPtr>> &out_control_peer_in_control_anchors,
      vector<vector<InDataAnchorPtr>> &out_control_peer_in_data_anchors);

  graphStatus TransOpFuse(const ComputeGraphPtr &graph);

  bool OpAccuracyAbilityCheck(const OpDescPtr &op_desc);

  graphStatus GetSubGraphsBetweenNormalNode(
      const OutDataAnchorPtr &out_anchor,
      vector<vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>>
  >& sub_graphs_out,
  vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> &nodes_list
  );

  graphStatus GetSubGraphNodesInfo();

  void GetControlAnchors();

  graphStatus InsertNewTransOp(const ComputeGraphPtr &graph, const OpDescPtr &cast_op,
                               const OpDescPtr &format_transfer_op, const int index,
                               const bool insert_cast_first);

  void EraseInvalidAnchorsPair();

  graphStatus RelinkNodesWhenDescNotChanged(const pair<OutDataAnchorPtr, InDataAnchorPtr> &begin_anchors_pair,
                                            const pair<OutDataAnchorPtr, InDataAnchorPtr> &end_anchors_pair,
                                            const int index);

  OpDescPtr GetFormatTransferOp(const GeTensorDesc &out_desc, const GeTensorDesc &in_desc);

  OpDescPtr GetCastOp(const GeTensorDesc &out_desc, const GeTensorDesc &in_desc);

  graphStatus TransOpFuseHandle(const ge::ComputeGraphPtr &graph, const int index);

  graphStatus AddTransNode(const ComputeGraphPtr &graph, const OpDescPtr &transop, NodePtr &trans_node);

  bool DescEqualCheck(ConstGeTensorDescPtr &desc_src, ConstGeTensorDescPtr &desc_dst) const;

  bool ShapeEqualCheck(const GeShape &src, const GeShape &dst) const;

  bool InsertCastFirstCheck(const GeTensorDesc &out_desc, const GeTensorDesc &in_desc) const;

  graphStatus RelinkControlEdge(const int index, const OutDataAnchorPtr &out_anchor,
                                const vector<NodePtr> &new_trans_nodes);

  graphStatus GetTransNode(const ComputeGraphPtr &graph,
                           const OpDescPtr &cast_op,
                           const OpDescPtr &format_transfer_op,
                           const bool insert_cast_first,
                           std::vector<NodePtr> &new_trans_nodes);

  void UpdateOutputName(const OutDataAnchorPtr &out_anchor, const InDataAnchorPtr &old_peer_in_anchor,
                        const NodePtr &in_owner_node);
  void UpdateInputName(const OutDataAnchorPtr &old_peer_out_anchor, const InDataAnchorPtr &in_anchor,
                       const NodePtr &out_owner_node);

  graphStatus RelinkControlEdgesWhenDescNotChanged(const pair<OutDataAnchorPtr, InDataAnchorPtr> &begin_anchors_pair,
                                                   const pair<OutDataAnchorPtr, InDataAnchorPtr> &end_anchors_pair,
                                                   const int index);

  graphStatus RelinkSubGraphControlEdges(const pair<OutDataAnchorPtr, InDataAnchorPtr> &begin_anchors_pair,
                                         const pair<OutDataAnchorPtr, InDataAnchorPtr> &end_anchors_pair,
                                         const int index);
  ///
  /// judge whether an operator is a transform op or not
  /// @param node
  /// @return True or False
  ///
  static bool IsTransOp(const NodePtr &node);

  static bool FusionFormatSupport(Format format);

  vector<vector<pair<OutDataAnchorPtr, InDataAnchorPtr>>>
  sub_graph_anchors_;
  vector<vector<NodePtr>> sub_graph_nodes_;
  vector<int> transop_num_count_;
  vector<bool> sub_graph_has_reshape_node_;
  vector<vector<OutControlAnchorPtr>> in_control_peer_out_control_anchors_;
  vector<vector<InControlAnchorPtr>> out_control_peer_in_control_anchors_;
  vector<vector<InDataAnchorPtr>> out_control_peer_in_data_anchors_;
  vector<vector<InControlAnchorPtr>> out_data_peer_in_control_anchors_;
  vector<bool> sub_graph_has_control_edge_;
  vector<bool> sub_graph_has_out_data_peer_in_control_edge_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_TRANSOP_WITHOUT_RESHAPE_FUSION_PASS_H_

