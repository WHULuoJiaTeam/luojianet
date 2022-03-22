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

#ifndef GE_GRAPH_PASSES_SAME_TRANSDATA_BREADTH_FUSION_PASS_H_
#define GE_GRAPH_PASSES_SAME_TRANSDATA_BREADTH_FUSION_PASS_H_

#include <utility>
#include <vector>
#include "inc/graph_pass.h"

namespace ge {
///
/// Transform operators depth fusion
///
class SameTransdataBreadthFusionPass : public GraphPass {
 public:
  SameTransdataBreadthFusionPass() {}
  virtual ~SameTransdataBreadthFusionPass() {}

  graphStatus Run(ComputeGraphPtr graph) override;

 private:
  graphStatus ExtractTransNode(const ComputeGraphPtr &graph);
  graphStatus GetSubGraphsBetweenNormalAndTransdataNode(OutDataAnchorPtr &out_anchor,
      std::vector<std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>>> &sub_graphs_out,
      std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> &nodes_list);

  void GetSubGraphNodesInfo();

  void EraseInvalidAnchorsPair();
  std::set<std::string> GetInControlIdentityNodes(const NodePtr &node, int subgraph_index);
  OpDescPtr GetCastOp(const GeTensorDesc &in_desc, const GeTensorDesc &out_desc);

  graphStatus AddCastNode(const ComputeGraphPtr &graph,
                          int anchors_index,
                          OutDataAnchorPtr &pre_out_anchor,
                          NodePtr &first_link_node);

  void GetSameTransdataNode(vector<int> &same_transdata_nodes);

  graphStatus ReLinkTransdataOutput2PreNode(const NodePtr &transdata_node, const OutDataAnchorPtr &pre_out_anchor,
                                            const NodePtr &relink_node);

  graphStatus RelinkTransdataControlEdge(ComputeGraphPtr graph,
                                         NodePtr transdata_node_remove,
                                         NodePtr transdata_node_keep);

  graphStatus LinkNewCastNode2RemainTransdata(const ComputeGraphPtr &graph,
                                              const vector<int> &same_transdata_nodes,
                                              const OutDataAnchorPtr &transdata_out_anchor,
                                              const NodePtr &transdata_node_keep);

  void UpdateTransdataDesc(const InDataAnchorPtr &transdata_in_anchor, const OpDescPtr &transdata_op_desc,
                           const ConstGeTensorDescPtr &head_output_desc);

  graphStatus RelinkRemainTransdata(const ComputeGraphPtr &graph, const vector<int> &same_transdata_nodes);

  graphStatus ReLinkTransdataControlOutput2PreNode(const NodePtr &transdata_node_keep,
                                                   const OutDataAnchorPtr &pre_out_anchor,
                                                   const OutControlAnchorPtr &transdata_peer_out_control_anchor);

  graphStatus ReuseNodesBeforeTransdata(int anchors_index, const OutDataAnchorPtr &transdata_out_anchor,
                                        NodePtr &relink_node);

  bool AllNodeBeforeTransdataHasOneDataOut(int anchors_index);

  graphStatus RelinkInControlEdge(const NodePtr &node_src, const NodePtr &node_dst);

  graphStatus ReLinkDataOutput2PreNode(const NodePtr &transdata_node,
                                       const OutDataAnchorPtr &pre_out_anchor,
                                       const NodePtr &relink_node);

  graphStatus ReLinkOutDataPeerInControlNodes2PreNode(const NodePtr &transdata_node,
                                                      const OutDataAnchorPtr &pre_out_anchor);

  void InsertSameTransdataNodeIndex(int anchors_index, vector<int> &same_transdata_nodes);

  graphStatus ReLinkOutControlPeerInControlAnchors(const NodePtr &transdata_node_keep,
                                                   const OutDataAnchorPtr &pre_out_anchor,
                                                   const OutControlAnchorPtr &transdata_peer_out_control_anchor);

  graphStatus ReLinkOutControlPeerInDataAnchors(const NodePtr &transdata_node_keep,
                                                const OutDataAnchorPtr &pre_out_anchor,
                                                const OutControlAnchorPtr &transdata_peer_out_control_anchor);

  void CopyTensorDesc(const ConstGeTensorDescPtr &src_desc, GeTensorDesc &dst_desc);

  ///
  /// judge whether an operator is a transform op or not
  /// @param node
  /// @return True or False
  ///
  static bool IsTransOp(const NodePtr &node);

  static bool IsHandleOp(const NodePtr &node);

  vector<vector<pair<OutDataAnchorPtr, InDataAnchorPtr>>> sub_graph_anchors_;
  vector<vector<NodePtr>> before_transdata_nodes_;
  vector<pair<int, InDataAnchorPtr>> all_transdata_nodes_;
  vector<vector<NodePtr>> sub_graph_nodes_;
  vector<int> transop_num_count_;
  vector<bool> sub_graph_has_reshape_node_;
  vector<vector<OutControlAnchorPtr>> peer_out_control_anchors_;
  vector<vector<InControlAnchorPtr>> peer_in_control_anchors_;
  vector<bool> sub_graph_has_control_edge_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_SAME_TRANSDATA_BREADTH_FUSION_PASS_H_
