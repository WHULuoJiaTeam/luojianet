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

#ifndef GE_GRAPH_PASSES_END_OF_SEQUENCE_ADD_CONTROL_EDGE_PASS_H_
#define GE_GRAPH_PASSES_END_OF_SEQUENCE_ADD_CONTROL_EDGE_PASS_H_

#include "external/graph/graph.h"
#include "inc/graph_pass.h"

namespace ge {
class EndOfSequenceAddControlPass : public GraphPass {
 public:
  EndOfSequenceAddControlPass() {}
  EndOfSequenceAddControlPass(const EndOfSequenceAddControlPass &eos_pass) = delete;
  EndOfSequenceAddControlPass &operator=(const EndOfSequenceAddControlPass &eos_pass) = delete;

  ~EndOfSequenceAddControlPass() override {}

  Status Run(ComputeGraphPtr graph) override;

 private:
  /**
  * Get EndOfSequence node in graph, nullptr if not exist.
  * @param graph
  * @return EndOfSequence node
  */
  inline NodePtr GetEndOfSequence(const ComputeGraphPtr &graph) const;
  /**
  * Check whether this node is a data-like node.
  * @param node
  * @return
  */
  bool IsDataLikeNode(const NodePtr &node);
  /**
  * Check whether this node is a data-like node.
  * @param node
  * @return
  */
  Status AddControlEdge(NodePtr &end_of_sequence, std::vector<NodePtr> &target_nodes);
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_END_OF_SEQUENCE_ADD_CONTROL_EDGE_PASS_H_
