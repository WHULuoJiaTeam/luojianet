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
#ifndef GE_GRAPH_PASSES_COND_PASS_H
#define GE_GRAPH_PASSES_COND_PASS_H

#include "graph/passes/base_pass.h"

namespace ge {
class CondPass : public BaseNodePass {
 public:
  Status Run(NodePtr &node) override;

 private:
  ///
  /// @brief Get cond info for if / while
  /// @param [in] node: If / While op
  /// @param [out] graph: owner_graph of if node / while_cond subgraph
  /// @param [out] peer_out_anchor: peer_cond_anchor
  /// @param [out] cond_in_anchor: cond_input
  /// @return Status
  ///
  static Status GetCondInfo(const NodePtr &node, ComputeGraphPtr &graph, OutDataAnchorPtr &peer_out_anchor,
                            InDataAnchorPtr &cond_in_anchor);

  ///
  /// @brief Get cond info for if node
  /// @param [in] node: If op
  /// @param [out] graph: owner_graph of if node
  /// @param [out] peer_out_anchor: peer_cond_anchor
  /// @param [out] cond_in_anchor: cond_input of if
  /// @return Status
  ///
  static Status GetCondInfoForIf(const NodePtr &node, ComputeGraphPtr &graph, OutDataAnchorPtr &peer_out_anchor,
                                 InDataAnchorPtr &cond_in_anchor);

  ///
  /// @brief Get cond info for while node
  /// @param [in] node: While op
  /// @param [out] graph: while_cond subgraph
  /// @param [out] peer_out_anchor: peer_cond_anchor
  /// @param [out] cond_in_anchor: input of NetOutput in cond_graph
  /// @return Status
  ///
  static Status GetCondInfoForWhile(const NodePtr &node, ComputeGraphPtr &graph, OutDataAnchorPtr &peer_out_anchor,
                                    InDataAnchorPtr &cond_in_anchor);

  ///
  /// @brief Process Cond Op with non-scalar cond_input
  /// @param [in] graph
  /// @param [in] peer_out_anchor: peer_cond_anchor
  /// @param [in] cond_in_anchor: cond_input
  /// @return Status
  ///
  Status HandleNonScalarCond(const ComputeGraphPtr &graph, const OutDataAnchorPtr &peer_out_anchor,
                             const InDataAnchorPtr &cond_in_anchor);

  ///
  /// @brief Process Cond Op with scalar-string cond_input
  /// @param [in] graph
  /// @param [in] peer_out_anchor: peer_cond_anchor
  /// @param [in] cond_in_anchor: cond_input
  /// @return Status
  ///
  Status HandleStringCond(const ComputeGraphPtr &graph, const OutDataAnchorPtr &peer_out_anchor,
                          const InDataAnchorPtr &cond_in_anchor);

  ///
  /// @brief Process Cond Op with scalar cond_input
  /// @param [in] graph
  /// @param [in] peer_out_anchor: peer_cond_anchor
  /// @param [in] cond_in_anchor: cond_input
  /// @param [in] src_type
  /// @return Status
  ///
  Status HandleScalarCond(const ComputeGraphPtr &graph, const OutDataAnchorPtr &peer_out_anchor,
                          const InDataAnchorPtr &cond_in_anchor, DataType src_type);

  ///
  /// @brief Insert node
  /// @param [in] graph
  /// @param [in] peer_out_anchor
  /// @param [in] in_data_anchor
  /// @param [in] type
  /// @return Status
  ///
  Status InsertNode(const ComputeGraphPtr &graph, const OutDataAnchorPtr &peer_out_anchor,
                    const InDataAnchorPtr &in_data_anchor, const std::string &type);

  ///
  /// @brief Add cast node
  /// @param [in] graph
  /// @param [in] name
  /// @param [in] tensor
  /// @param [in] src
  /// @param [in] dst
  /// @return NodePtr
  ///
  NodePtr AddCastNode(const ComputeGraphPtr &graph, const std::string &name, const GeTensorDesc &tensor,
                      DataType src, DataType dst);
};
}  // namespace ge
#endif //GE_GRAPH_PASSES_COND_PASS_H
