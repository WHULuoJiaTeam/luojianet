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

#ifndef GE_GRAPH_PASSES_TRANSOP_DEPTH_FUSION_PASS_H_
#define GE_GRAPH_PASSES_TRANSOP_DEPTH_FUSION_PASS_H_

#include <stack>
#include <string>
#include <vector>

#include "inc/graph_pass.h"

namespace ge {
///
/// Transform operators depth fusion
///
class TransOpDepthFusionPass : public GraphPass {
 public:
  TransOpDepthFusionPass() = default;
  ~TransOpDepthFusionPass() = default;

  graphStatus Run(ge::ComputeGraphPtr graph) override;

 private:
  ///
  /// Judge whether the node can be deleted in depth fusion
  /// @param node
  /// @return True or False
  ///
  static bool CheckNodeCanBeDeleted(const NodePtr &node);

  ///
  /// two transform nodes can be offset only when the front node's input is
  /// consistent with the back one's output
  /// @param src_node: the front node
  /// @param dst_node: the back node
  /// @return True or False, whether can be offset or not
  ///
  static bool DescAreSymmetry(const NodePtr &src_node, const NodePtr &dst_node);

  ///
  /// update the input_name and src_name info when the relationship was changed
  /// @param src_out_anchor: the new peer in data anchor of dst_in_anchor
  /// @param old_src_anchor: the original peer in data anchor of dst_in_anchor
  /// @param dst_in_anchor: the target anchor
  /// @return Status
  ///
  static graphStatus UpdateSrcAttr(const OutDataAnchorPtr &src_out_anchor, const OutDataAnchorPtr &old_src_anchor,
                                   const InDataAnchorPtr &dst_in_anchor);

  ///
  /// Depth-first recursive to traverse all the transops
  /// @param dst_in_anchor: each in_data_anchor is set as the root in the recursive
  /// @return Status
  ///
  graphStatus RecursiveInDepth(const InDataAnchorPtr &dst_in_anchor, const ge::ComputeGraphPtr &graph);

  ///
  /// Remove transop by using interface: IsolateNode & RemoveNodeWithoutRelink
  /// @param node: the trans op which will be removed
  /// @return Status
  ///
  static graphStatus RemoveNode(const NodePtr &node, const ge::ComputeGraphPtr &graph);

  ///
  /// Relink the offset trans op with it's former's father node.
  /// Note: control edge will be added to link the two offset ops, if the former
  /// trans op have in control nodes
  /// @param new_out_anchor: out_data_anchor of father node of the former trans op
  /// @param old_out_anchor: out_data_anchor of the former trans op
  /// @param in_data_anchor: in_data_anchor of the after trans op
  /// @return Status
  ///
  static graphStatus RelinkEdges(const OutDataAnchorPtr &new_out_anchor, const OutDataAnchorPtr &old_out_anchor,
                                 const InDataAnchorPtr &in_data_anchor);

  ///
  /// @param trans_op_  : the trans op which can't be offset at the moment
  /// @param offset_op_ : the former one of the offset pair nodes
  ///
  std::stack<NodePtr> trans_op_;
  std::stack<NodePtr> offset_op_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_TRANSOP_DEPTH_FUSION_PASS_H_
