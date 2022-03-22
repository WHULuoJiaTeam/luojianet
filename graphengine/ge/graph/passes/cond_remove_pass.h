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
#ifndef GE_GRAPH_PASSES_COND_REMOVE_PASS_H
#define GE_GRAPH_PASSES_COND_REMOVE_PASS_H

#include "graph/passes/base_pass.h"

namespace ge {
class CondRemovePass : public BaseNodePass {
 public:
  Status Run(NodePtr &node) override;

 private:
  ///
  /// @brief Get cond info for if/case node
  /// @param [in] node: If/Case op
  /// @param [out] graph: owner_graph of if node
  /// @param [out] cond_out_anchor: peer_cond_anchor
  /// @param [out] cond_in_anchor: cond_input of if
  /// @return Status
  ///
  Status GetCondInfoForIfCase(const NodePtr &node, ComputeGraphPtr &graph, OutDataAnchorPtr &cond_out_anchor,
                              InDataAnchorPtr &cond_in_anchor);
  ///
  /// @brief Get cond info for if/case node
  /// @param [in] node: If/Case op
  /// @param [out] graph: owner_graph of if node
  /// @param [out] cond_out_anchor: peer_cond_anchor
  /// @param [out] cond_in_anchor: cond_input of if
  /// @return Status
  ///
  Status GetCondInfo(const NodePtr &node, ComputeGraphPtr &graph, OutDataAnchorPtr &cond_out_anchor,
                     InDataAnchorPtr &cond_in_anchor);
  ///
  /// @brief Check if condition input is const, for if / case / while
  ///
  bool CheckIfCondConstInput(const OutDataAnchorPtr &cond_out_anchor, const InDataAnchorPtr &cond_in_anchor,
                             int32_t &cond_index);

  ///
  /// @brief Remove if dead branch, for if
  ///
  Status GetIfChosenBranch(const NodePtr &node, const uint32_t cond_index, ComputeGraphPtr &compute_graph);

  ///
  /// @brief Remove if dead branch, for case
  ///
  Status GetCaseChosenBranch(const NodePtr &node, const uint32_t cond_index, ComputeGraphPtr &compute_graph);

  ///
  /// @brief Remove dead condition input, for if / case / while
  ///
  Status RemoveDeadCondLink(const int32_t index, const NodePtr &node);

  ///
  /// @brief Remove if dead branch, for if
  ///
  Status ReplaceIfCaseNodeWithPartitioncall(const NodePtr &node, const ComputeGraphPtr &save_branch);

  OpDescPtr CreateSubgraphOpDesc(const NodePtr &node, const std::string &name, size_t input_num, size_t output_num);

  int32_t GetCondIndex(const ConstGeTensorPtr &tensor);
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_COND_REMOVE_PASS_H
