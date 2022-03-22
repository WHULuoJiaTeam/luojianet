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

#ifndef GE_GRAPH_PASSES_ATOMIC_ADDR_CLEAN_PASS_H_
#define GE_GRAPH_PASSES_ATOMIC_ADDR_CLEAN_PASS_H_

#include <vector>

#include "external/graph/graph.h"
#include "inc/graph_pass.h"

namespace ge {
/*
 * Atomic addr clean task fusion
 * Find all atomic op in graph,and insert one AtomicAddrClean op.
 * To clean atomic output and workspace once for all.
 * before iteration starts, empty AtomicAdd output, workspace memory
 *         op1                         op1
 *          |                           |
 *         op2(atomic)       ==>       op2
 *          |                           |  \
 *         op3(atomic)                 op3 -AtomicClean
 */
class AtomicAddrCleanPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph);
  Status ClearStatus() override;

 private:
  /**
    * HandleLoopGraph
    * @param graph
    * @return
    */
  Status HandleLoopGraph(ComputeGraphPtr &graph, const vector<NodePtr> &atomic_node_vec);
  /**
   * HandleNormalGraph
   * @param graph
   * @return
   */
  Status HandleNormalGraph(ComputeGraphPtr &graph, const vector<NodePtr> &atomic_node_vec);
  /**
   * Insert atomic clean node to graph
   * @param graph
   * @return
   */
  NodePtr InsertAtomicAddrCleanNode(ComputeGraphPtr &graph);

  /**
   * Link control anchor from atomic clean node to atomic node
   * @param atomic_node
   * @param atomic_clean_node
   * @return
   */
  Status LinkToAtomicNode(const NodePtr &atomic_node, NodePtr &atomic_clean_node);

  /**
   * Link atomic clean node to all potential precedence nodes which may execute before atomic clean node
   * @param graph
   * @param atomic_clean_node
   * @return
   */
  Status LinkToPotentialPrecedenceNode(ComputeGraphPtr &graph, NodePtr &atomic_clean_node,
                                       const std::vector<NodePtr> &dispersed_atomic_nodes);

  /**
   * Check if this node is atomic op.
   * @param node
   * @return
   */
  bool IsAtomicOp(const NodePtr &node);

  /**
   * Handle atomic node in unknown graph
   * @param atomic_node_vec: atomic node vector in unknown graph
   * @return
   */
  Status CompileUnknownGraphOp(const vector<NodePtr> &atomic_node_vec);

  Status HandleDispersedAtomicNodes(ComputeGraphPtr &graph, const std::vector<NodePtr> &atomic_node_vec,
                                    std::vector<NodePtr> &common_atomic_nodes,
                                    std::vector<NodePtr> &dispersed_atomic_nodes);

  bool CheckAtomicFromOpsKernel(const NodePtr &node);

  bool IsOutputIndexPeerInputAtomic(const NodePtr &node, int64_t output_index);

  bool CheckSkipInsertInLoopGraph(const NodePtr &node);

  vector<NodePtr> hcom_node_vec_;
  bool is_loop_graph_ = false;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_ATOMIC_ADDR_CLEAN_PASS_H_
