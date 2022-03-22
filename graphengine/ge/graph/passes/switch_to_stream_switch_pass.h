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

#ifndef GE_GRAPH_PASSES_SWITCH_TO_STREAM_SWITCH_PASS_H_
#define GE_GRAPH_PASSES_SWITCH_TO_STREAM_SWITCH_PASS_H_

#include "inc/graph_pass.h"

namespace ge {
/* Variable Initialize Flow, take as FrameworkOp
                  +-----------+
                  |   Merge   |
                  +-----------+
                  /           \
                0/             \x
                /               \
     +-----------+             +-----------+
     |  Switch   |             |  Switch   |
     +-----------+             +-----------+
      |         |F             T|         |
     0|         |               |        x|
      |         |               |         |
      |     +-----------------------+     |
      |     | IsVariableInitialized |     |
      |     +-----------------------+     |
      |                         |         |
      |                         |         |
      |                         |         |
  +-----------+                +-----------+
  |   Const   |                | VariableV2|
  +-----------+                +-----------+


  Switch branch op optimize, Switches in same case merge to one StreamSwitch, update following nodes' input

                                            +-----------+
                                          / |   task2   | \
                                        T/  +-----------+  \
        +-----------+     +-----------+ /                   \ +-----------+     +-----------+
        |   task1   | --> |  Switch   |                       |   task4   | --> |   noop    |
        +-----------+     +-----------+ \                   / +-----------+     +-----------+
                                        F\  +-----------+  /
                                          \ |   task3   | /
                                            +-----------+

                cond(x < y, lambda: add(x, z), lambda: square(y))

                    +-----------+                                                 +-----------+
                    |   Merge   |                                    +------------|StreamMerge|----------+
                    +-----------+                                    |            +-----------+          |
                    /           \                                    |                 |                 |
                   /             \                                   |c                |                 |c
                  /               \                             +----------+      -----------      +----------+
        +-----------+           +-----------+                   | Active_f |     /           \     | Active_t |
        |  Square   |           |    Add    |                   +----------+    /             \    +----------+
        +-----------+           +-----------+                         \        /               \       /
              /                  /         \                           \c     /                 \     /c
            y/                 x/           \z                        +-----------+         +-----------+
            /                  /             \                        |  Square   |         |    Add    |
   +-----------+     +-----------+        +-----------+               +-----------+         +-----------+
   |  Switch   |     |  Switch   |        |  Switch   |  ====>            /   |               /   |   \
   +-----------+     +-----------+        +-----------+                  /    |              /    |    \
    y|       |F       T|       |x          T|       |z            +--------+  |       +--------+  |  +--------+
     |       |         |       |            |       |             | y/read |  |       | x/read |  |  | z/read |
     |      +-----------+      |            |       |             +--------+  |       +--------+  |  +--------+
     |      |   Less    |-------------------+       |                         |c                  |c
     |      +-----------+      |                    |               +----------------+     +----------------+
     |                         |                    |               | StreamSwitch_f |     | StreamSwitch_t |
     |                         |                    |               +----------------+     +----------------+
 +-----------+         +-----------+      +-----------+                    |                      |
 |  y/read   |         |  x/read   |      |  z/read   |                    |     +-----------+    |
 +-----------+         +-----------+      +-----------+                    +-----|   Less    |----+
                                                                                 +-----------+
*/
class SwitchToStreamSwitchPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph);

  ///
  /// @brief Clear Status, used for subgraph pass
  /// @return
  ///
  Status ClearStatus() override;

 private:
  ///
  /// @brief Check cyclic dependence
  /// @param [in] graph
  /// @return Status
  ///
  Status CheckCycleDependence(const ComputeGraphPtr &graph);

  ///
  /// @brief Mark cyclic dependence
  /// @param [in] graph
  /// @param [in] cond_switch_map
  /// @return void
  ///
  void MarkCycleDependence(const std::unordered_map<NodePtr, std::vector<NodePtr>> &cond_switch_map);

  ///
  /// @brief Replace Switch Op
  /// @param [in] graph
  /// @param [in] switch_node
  /// @return Status
  ///
  Status ReplaceSwitchNode(const ComputeGraphPtr &graph, const NodePtr &switch_node);

  ///
  /// @brief Bypass Switch Node
  /// @param [in] switch_node
  /// @param [out] peer_data_anchor
  /// @param [out] peer_cond_anchor
  /// @return Status
  ///
  Status BypassSwitchNode(const NodePtr &switch_node, OutDataAnchorPtr &peer_data_anchor,
                          OutDataAnchorPtr &peer_cond_anchor);

  ///
  /// @brief Find Switch cond input
  /// @param [out] peer_cond_anchor
  /// @return Status
  ///
  Status FindSwitchCondInput(OutDataAnchorPtr &peer_cond_anchor);

  ///
  /// @brief Create StreamSwitch Node
  /// @param [in] graph
  /// @param [in] switch_node
  /// @param [in] suffix
  /// @param [in] peer_cond_anchor
  /// @return ge::NodePtr
  ///
  NodePtr CreateStreamSwitchNode(const ComputeGraphPtr &graph, const NodePtr &switch_node, const std::string &suffix,
                                 const OutDataAnchorPtr &peer_cond_anchor);

  ///
  /// @brief Mark Switch Branch
  /// @param [in] peer_cond_anchor
  /// @param [in] stream_switch
  /// @param [in] true_branch_flag
  /// @return Status
  ///
  Status MarkBranches(const OutDataAnchorPtr &peer_cond_anchor, const NodePtr &stream_switch_node,
                      bool true_branch_flag);

  ///
  /// @brief Get group_id for switch_node
  /// @param [in] node
  /// @return group_id
  ///
  int64_t GetGroupId(const NodePtr &node);

  ///
  /// @brief Combine switch nodes link to same cond
  /// @param [in] graph
  /// @return Status
  ///
  Status CombineSwitchNode(const ComputeGraphPtr &graph);

  ///
  /// @brief Create cast node
  /// @param [in] graph
  /// @param [in] peer_cond_anchor
  /// @return NodePtr
  ///
  NodePtr CreateCastOp(const ComputeGraphPtr &graph, const OutDataAnchorPtr &peer_cond_anchor);

  ///
  /// @brief Create Active Op
  /// @param [in] graph
  /// @param [in] cond_node
  /// @return ge::NodePtr
  ///
  NodePtr CreateActiveNode(const ComputeGraphPtr &graph, const NodePtr &node);

  ///
  /// @brief Add const node as switch input1
  /// @param [in] graph
  /// @param [in] stream_switch
  /// @return Status
  ///
  Status AddConstNode(const ComputeGraphPtr &graph, const NodePtr &stream_switch_node);

  ///
  /// @brief Modify in ctl edge for switch_node
  /// @param [in] switch_node
  /// @param [in] cast_node
  /// @param [in] same_cond_switch
  /// @return Status
  ///
  Status ModifySwitchInCtlEdges(const NodePtr &switch_node, const NodePtr &cast_node,
                                const std::set<NodePtr> &same_cond_switch);

  ///
  /// @brief Modify out ctl edge for switch_node
  /// @param [in] switch_node
  /// @param [in] stream_switch
  /// @param [in] active_node
  /// @return Status
  ///
  Status ModifySwitchOutCtlEdges(const NodePtr &switch_node, const NodePtr &stream_switch, const NodePtr &active_node);

  ///
  /// @brief Check duplicate node_name
  /// @param [in] node_name
  /// @return std::string
  ///
  std::string CheckDuplicateName(const std::string &node_name);

  ///
  /// @brief Move Control Edges
  /// @param [in] old_node
  /// @param [in] new_node
  /// @return void
  ///
  void MoveCtrlEdges(const NodePtr &old_node, const NodePtr &new_node);

  std::vector<NodePtr> switch_nodes_;
  std::unordered_map<NodePtr, std::set<std::string>> switch_cyclic_map_;
  std::set<NodePtr> bypass_nodes_;
  std::vector<NodePtr> stream_switch_nodes_;
  std::unordered_map<OutDataAnchorPtr, std::map<int64_t, std::vector<std::list<NodePtr>>>> cond_node_map_;
  std::unordered_map<NodePtr, std::set<std::string>> switch_node_map_;
  std::map<std::string, uint32_t> node_num_map_;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_SWITCH_TO_STREAM_SWITCH_PASS_H_
