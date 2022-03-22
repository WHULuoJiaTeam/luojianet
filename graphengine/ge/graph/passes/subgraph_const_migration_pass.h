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

#ifndef GE_COMMON_SUBGRAPH_CONST_MIGRATION_H_
#define GE_COMMON_SUBGRAPH_CONST_MIGRATION_H_

#include "external/graph/types.h"
#include "inc/graph_pass.h"

#include <map>
#include <set>
#include <vector>
#include <string>

using std::set;
using std::map;

namespace ge {
class SubgraphConstMigrationPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph) override;

 private:
  ///
  /// @ingroup ge
  /// @brief Get all Const/Data nodes for all subgraph.
  /// @param [in] graph: Root compute graph.
  /// @param [in] func_desc: functional OpDesc of Case.
  /// @param [out] all_const_nodes: Const groups of subgraph.
  /// @param [out] all_data_nodes: Data groups of subgraph.
  /// @return 0: SUCCESS / others: FAILED
  ///
  Status ClassifyGraphNodes(const ComputeGraphPtr &graph, const OpDescPtr &func_desc,
                            map<ComputeGraphPtr, map<string, NodePtr>> &all_const_nodes,
                            map<ComputeGraphPtr, map<uint32_t, NodePtr>> &all_data_nodes);

  ///
  /// @ingroup ge
  /// @brief Get parent_index for Const node migration.
  /// @param [in] all_data_nodes: Data groups of subgraph.
  /// @param [in] const_node: Const node will process.
  /// @param [out] parent_index: parent index for replace Data.
  /// @return true: SUCCESS / false: FAILED
  ///
  bool GetAssociatedNodes(const map<ComputeGraphPtr, map<uint32_t, NodePtr>> &all_data_nodes,
                          const NodePtr &const_node, uint32_t &parent_index);

  ///
  /// @ingroup ge
  /// @brief Check parallel node is same for all subgraph.
  /// @param [in] all_const_nodes: Const groups of subgraph.
  /// @param [in] const_node: Const Node for migration.
  /// @param [in] node_key: Key of Const node.
  /// @return true: Same / false: not same
  ///
  bool IsParallelNodeSame(const map<ComputeGraphPtr, map<string, NodePtr>> &all_const_nodes,
                          const NodePtr &const_node, const string &node_key);

  ///
  /// @ingroup ge
  /// @brief Migration subgraph Node to Root
  /// @param [in] graph: Root compute graph.
  /// @param [in] func_node: functional Node of Case.
  /// @param [in] all_const_nodes: Const groups of subgraph.
  /// @param [in] all_data_nodes: Data groups of subgraph.
  /// @param [in] const_node: Const Node for migration.
  /// @param [in] node_key: Key of Const node for migration.
  /// @return 0: SUCCESS / others: FAILED
  ///
  Status GraphNodeMigration(const ComputeGraphPtr &graph, const NodePtr &func_node,
                            const map<ComputeGraphPtr, map<string, NodePtr>> &all_const_nodes,
                            map<ComputeGraphPtr, map<uint32_t, NodePtr>> &all_data_nodes,
                            const NodePtr &const_node, const string &node_key);

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
                          const map<ComputeGraphPtr, map<string, NodePtr>> &all_const_nodes,
                          const map<ComputeGraphPtr, map<uint32_t, NodePtr>> &all_data_nodes,
                          const string &node_key, uint32_t parent_index);

  ///
  /// @ingroup ge
  /// @brief Append Input Tensor for functional node.
  /// @param [in] graph_nodes: Const groups of subgraph.
  /// @param [in/out] parent_index: Parent index for migration.
  /// @param [in/out] all_data_nodes: Data groups of subgraph.
  /// @return 0: SUCCESS / others: FAILED
  ///
  Status AppendParallelNode(const NodePtr &func_node, uint32_t &parent_index,
                            map<ComputeGraphPtr, map<uint32_t, NodePtr>> &all_data_nodes);

  ///
  /// @ingroup ge
  /// @brief Delete Node from subgraph.
  /// @param [in] graph: subgraph for process.
  /// @param [in] const_node: Node will move to parent.
  /// @param [in] data_node: Place holder for Const.
  /// @return 0: SUCCESS / others: FAILED
  ///
  Status DetachParallelNode(const ComputeGraphPtr &graph, const NodePtr &const_node, const NodePtr &data_node);

  ///
  /// @ingroup ge
  /// @brief Move Node to Parent Graph.
  /// @param [in] graph: Parent compute graph.
  /// @param [in] func_node: functional Node of Case.
  /// @param [in] const_node: Node will move to parent.
  /// @param [in] parent_index: Parent index of Node input.
  /// @return 0: SUCCESS / others: FAILED
  ///
  Status AttachParallelNode(const ComputeGraphPtr &graph, const NodePtr &func_node,
                            const NodePtr &const_node, uint32_t parent_index);

  void GetPeerNameList(const NodePtr &node, set<string> &peer_name_list);
};
}  // namespace ge
#endif  // GE_COMMON_SUBGRAPH_CONST_MIGRATION_H_