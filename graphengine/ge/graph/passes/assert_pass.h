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


#ifndef GE_GRAPH_PASSES_ASSERT_PASS_H_
#define GE_GRAPH_PASSES_ASSERT_PASS_H_

#include <vector>

#include "graph/passes/base_pass.h"

namespace ge {
class AssertPass : public BaseNodePass {
 public:
  Status Run(NodePtr& node) override;

 private:
  ///
  /// collect assert and other unused ops
  /// @param assert_node assert node
  /// @param nodes_unused nodes to be deleted
  /// @return void
  ///
  void CollectUnusedNode(const NodePtr &assert_node, std::vector<ge::NodePtr>& nodes_unused);

  ///
  /// remove unused nodes from graph
  /// @param graph
  /// @param nodes_unused nodes to be deleted
  /// @return Status
  ///
  Status RemoveUnusedNode(std::vector<NodePtr>& nodes_unused);
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_ASSERT_PASS_H_
