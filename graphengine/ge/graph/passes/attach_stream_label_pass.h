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

#ifndef GE_GRAPH_PASSES_ATTACH_STREAM_LABEL_PASS_H_
#define GE_GRAPH_PASSES_ATTACH_STREAM_LABEL_PASS_H_

#include <stack>
#include "inc/graph_pass.h"

namespace ge {
class AttachStreamLabelPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph);

 private:
  ///
  /// @brief Find StreamSwitch / StreamMerge / Enter node
  /// @param [in] graph
  /// @param [out] need_label_nodes
  /// @param [out] enter_nodes
  /// @param [out] branch_head_nodes
  /// @return void
  ///
  void FindNodes(const ComputeGraphPtr &graph, std::vector<NodePtr> &need_label_nodes,
                 std::vector<NodePtr> &enter_nodes, std::map<NodePtr, NodePtr> &branch_head_nodes);

  ///
  /// @brief update cond branch
  /// @param [in] node
  /// @param [in] branch_head_nodes
  /// @return Status
  ///
  Status UpdateCondBranch(const NodePtr &node, const std::map<NodePtr, NodePtr> &branch_head_nodes);

  ///
  /// @brief attach flag
  /// @param [in] node
  /// @param [out] stream_label
  /// @return Status
  ///
  static Status AttachFlag(const NodePtr &node, std::string &stream_label);

  ///
  /// @brief Update stream_label for loop_branch
  /// @param [in] enter_nodes
  /// @param [in] stream_label
  /// @return Status
  ///
  static Status UpdateLoopBranch(const std::stack<NodePtr> &enter_nodes, const std::string &stream_label);

  ///
  /// @brief Update stream_label start with enter nodes
  /// @param [in] enter_nodes
  /// @return Status
  ///
  Status UpdateEnterNode(const std::vector<NodePtr> &enter_nodes);

  ///
  /// @brief Set stream_label for enter_nodes
  /// @param [in] enter_nodes
  /// @param [in] active_node
  /// @return Status
  ///
  static Status SetEnterLabel(const std::vector<NodePtr> &enter_nodes, const NodePtr &active_node);
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_ATTACH_STREAM_LABEL_PASS_H_
