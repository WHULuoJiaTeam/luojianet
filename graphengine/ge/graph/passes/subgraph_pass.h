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

#ifndef GE_GRAPH_PASSES_SUBGRAPH_PASS_H_
#define GE_GRAPH_PASSES_SUBGRAPH_PASS_H_

#include "inc/graph_pass.h"

namespace ge {
class SubgraphPass : public GraphPass {
 public:
  /**
   * @ingroup ge
   * @brief Subgraph optimizer.
   * @param [in] graph: Input ComputeGraph
   * @return: 0 for success / others for fail
   */
  Status Run(ComputeGraphPtr graph) override;

 private:
  /**
   * @ingroup ge
   * @brief Check Subgraph Data node.
   * @param [in] graph: ComputeGraph.
   * @param [in] node: NetOutput node in Subgraph.
   * @return: 0 for SUCCESS / others for FAILED
   */
  Status SubgraphInputNode(const ComputeGraphPtr &graph, const NodePtr &node);

  /**
   * @ingroup ge
   * @brief Check Subgraph NetOutput node.
   * @param [in] graph: ComputeGraph.
   * @param [in] node: NetOutput node in Subgraph.
   * @return: 0 for SUCCESS / others for FAILED
   */
  Status SubgraphOutputNode(const ComputeGraphPtr &graph, const NodePtr &node);

  /**
   * @ingroup ge
   * @brief Check is Input->While and Input link to other nodes
   * @param [in] graph: ComputeGraph.
   * @param [in] node: While node.
   * @return: 0 for SUCCESS / others for FAILED
   */
  Status WhileInputNodes(const ComputeGraphPtr &graph, const NodePtr &node);

  /**
   * @ingroup ge
   * @brief Check body subgraph of While op
   * @param [in] graph: ComputeGraph.
   * @param [in] node: While node.
   * @return: 0 for SUCCESS / others for FAILED
   */
  Status WhileBodySubgraph(const ComputeGraphPtr &graph, const NodePtr &node);

  /**
   * @ingroup ge
   * @brief Insert input memcpy node in while_body
   * @param [in] graph: while_body
   * @param [in] data_nodes: data_nodes
   * @return: 0 for SUCCESS / others for FAILED
   */
  Status InsertInputMemcpy(const ComputeGraphPtr &graph, const std::vector<NodePtr> &data_nodes);

  /**
   * @ingroup ge
   * @brief Insert output memcpy node in while_body
   * @param [in] graph: while_body
   * @param [in] output_node: NetOutput
   * @param [in] bypass_index
   * @return: 0 for SUCCESS / others for FAILED
   */
  Status InsertOutputMemcpy(const ComputeGraphPtr &graph, const NodePtr &output_node,
                            const std::set<uint32_t> &bypass_index);

  /**
   * @ingroup ge
   * @brief Check is data->netoutput without change in while body
   * @param [in] node: data node
   * @param [out] bypass_index
   * @return: false for data->netoutput without change in while body / for true for others
   */
  bool CheckInsertInputMemcpy(const NodePtr &node, std::set<uint32_t> &bypass_index);

  /**
   * @ingroup ge
   * @brief Check is AtomicOp->NetOutput
   * @param [in] node
   * @param [in] out_index
   * @return: true for AtomicOp->NetOutput / false for others
   */
  bool IsAtomicRequired(const NodePtr &node, int64_t out_index);

  /**
   * @ingroup ge
   * @brief Check is OutputContinuesRequiredOp->NetOutput
   * @param [in] node
   * @return: true for OutputContinuesRequiredOp->NetOutput / false for others
   */
  bool IsOutputContinuesRequired(const NodePtr &node);

  /**
   * @ingroup ge
   * @brief Check is InputContinuesRequiredOp->NetOutput
   * @param [in] node
   * @return: true for InputContinuesRequiredOp->NetOutput / false for others
   */
  bool IsInputContinuesRequired(const NodePtr &node);

  /**
   * @ingroup ge
   * @brief Insert memcpy node
   * @param [in] graph
   * @param [in] out_anchor
   * @param [in] in_anchors
   * @param [in] name
   * @return: 0 for success / others for fail
   */
  Status InsertMemcpyNode(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_anchor,
                          const std::vector<InDataAnchorPtr> &in_anchors, const std::string &name);

  ///
  /// @brief Insert node: src->insert_node:input_index, insert_node:output_index->dst
  /// @param [in] src
  /// @param [in] dsts
  /// @param [in] insert_node
  /// @param [in] input_index
  /// @param [in] output_index
  /// @return Status
  ///
  Status InsertNodeBetween(const OutDataAnchorPtr &src, const std::vector<InDataAnchorPtr> &dsts,
                           const NodePtr &insert_node, uint32_t input_index, uint32_t output_index);
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_SUBGRAPH_PASS_H_
