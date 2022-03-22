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

#ifndef GE_COMMON_SUBEXPRESSION_MIGRATION_H_
#define GE_COMMON_SUBEXPRESSION_MIGRATION_H_

#include "external/graph/types.h"
#include "inc/graph_pass.h"

#include <map>
#include <set>
#include <vector>
#include <string>

using std::set;
using std::map;

namespace ge {
class SubexpressionMigrationPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph) override;

 private:
  ///
  /// @ingroup ge
  /// @brief Get all Data nodes for all subgraph.
  /// @param [in] graph: Root compute graph.
  /// @param [in] func_desc: functional OpDesc of Case.
  /// @param [out] graph_nodes: Data groups of subgraph.
  /// @return 0: SUCCESS / others: FAILED
  ///
  Status ClassifyDataNodes(const ComputeGraphPtr &graph, const OpDescPtr &func_desc,
                           map<ComputeGraphPtr, map<uint32_t, NodePtr>> &graph_nodes);

  ///
  /// @ingroup ge
  /// @brief Get all Data nodes for all subgraph.
  /// @param [in] node: Node Directly to Data.
  /// @param [out] inputs: parent index of Input.
  /// @param [out] outputs: parent index of Output.
  /// @return true: SUCCESS / false: FAILED
  ///
  bool GetAssociatedNodes(const NodePtr &node, map<uint32_t, uint32_t> &inputs, map<uint32_t, uint32_t> &outputs);

  ///
  /// @ingroup ge
  /// @brief Get all Data nodes for all subgraph.
  /// @param [in] graph_nodes: Data groups of subgraph.
  /// @param [in] base_node: Data Node for migration.
  /// @param [in] node_idx: Parent index of Data node.
  /// @param [in] anchor_idx: Anchor index of node.
  /// @return true: Same / false: not same
  ///
  bool IsParallelNodeSame(const map<ComputeGraphPtr, map<uint32_t, NodePtr>> &graph_nodes,
                          const NodePtr &base_node, uint32_t node_idx, uint32_t anchor_idx);

  ///
  /// @ingroup ge
  /// @brief Migration subgraph Node to Root
  /// @param [in] graph: Root compute graph.
  /// @param [in] func_node: functional Node of Case.
  /// @param [in] graph_nodes: Data groups of subgraph.
  /// @param [in] data_base: Data Node for migration.
  /// @param [in] data_idx: Data groups of subgraph.
  /// @return 0: SUCCESS / others: FAILED
  ///
  Status GraphNodeMigration(const ComputeGraphPtr &graph, const NodePtr &func_node,
                            map<ComputeGraphPtr, map<uint32_t, NodePtr>> &graph_nodes,
                            const NodePtr &data_base, uint32_t data_idx);

  ///
  /// @ingroup ge
  /// @brief Move node to Parent graph.
  /// @param [in] graph: Root compute graph.
  /// @param [in] func_node: functional Node of Case.
  /// @param [in] graph_nodes: Data groups of subgraph.
  /// @param [in] anchor_idx: anchor index of move Node.
  /// @param [in] inputs: Parent index of Node input.
  /// @param [in] outputs: Parent index of Node output.
  /// @return 0: SUCCESS / others: FAILED
  ///
  Status MoveNodeToParent(const ComputeGraphPtr &graph, const NodePtr &func_node,
                          const map<ComputeGraphPtr, map<uint32_t, NodePtr>> &graph_nodes, uint32_t anchor_idx,
                          const map<uint32_t, uint32_t> &inputs, const map<uint32_t, uint32_t> &outputs);

  ///
  /// @ingroup ge
  /// @brief Append Input Tensor for functional node.
  /// @param [in] graph_nodes: Data groups of subgraph.
  /// @param [in] func_node: functional Node of Case.
  /// @param [in] outputs: Parent index of Node output.
  /// @return 0: SUCCESS / others: FAILED
  ///
  Status AppendParallelNode(map<ComputeGraphPtr, map<uint32_t, NodePtr>> &graph_nodes,
                            const NodePtr &func_node, map<uint32_t, uint32_t> &outputs);

  ///
  /// @ingroup ge
  /// @brief Delete Node from all subgraph.
  /// @param [in] graph_nodes: Data groups of subgraph.
  /// @param [in] detach: Node will move to parent.
  /// @param [in] outputs: Parent index of Node output.
  /// @return 0: SUCCESS / others: FAILED
  ///
  Status DetachParallelNode(const map<uint32_t, NodePtr> &graph_datas, const NodePtr &detach,
                            const map<uint32_t, uint32_t> &outputs);

  ///
  /// @ingroup ge
  /// @brief Move Node to Parent Graph.
  /// @param [in] graph: Parent compute graph.
  /// @param [in] func_node: functional Node of Case.
  /// @param [in] attach: Node will move to parent.
  /// @param [in] inputs: Parent index of Node input.
  /// @param [in] outputs: Parent index of Node output.
  /// @return 0: SUCCESS / others: FAILED
  ///
  Status AttachParallelNode(const ComputeGraphPtr &graph, const NodePtr &func_node, const NodePtr &attach,
                            const map<uint32_t, uint32_t> &inputs, const map<uint32_t, uint32_t> &outputs);

  bool migration_append_{false};
};
}  // namespace ge
#endif  // GE_COMMON_SUBEXPRESSION_MIGRATION_H_
