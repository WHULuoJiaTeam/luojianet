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

#ifndef GE_GRAPH_PASSES_PARALLEL_GROUP_PASS_H
#define GE_GRAPH_PASSES_PARALLEL_GROUP_PASS_H

#include <map>
#include <unordered_set>
#include "external/graph/graph.h"
#include "inc/graph_pass.h"

namespace ge {
class ParallelGroupPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph) override;
 private:
  Status ProcessGraphGroupNodes(ComputeGraphPtr graph, int32_t depth, std::unordered_set<std::string> &parallel_group);

  Status AddCtrlEdge(NodePtr pre_node, NodePtr cur_node);

  Status ReplaceWithSwitchAndMerge(NodePtr pre_node, NodePtr cur_node,
                                   const std::map<NodePtr, std::pair<std::set<NodePtr>, NodePtr>> &node_2_switch_merge);

  bool HasSameSwitch(const std::set<NodePtr> &a, const std::set<NodePtr> &b);

  Status ProcessGroupNodeInSwitch(ComputeGraphPtr graph,
                                  std::map<NodePtr, std::pair<std::set<NodePtr>, NodePtr>> &node_2_switch_merge);

  void FindGroupNodeAndMerge(NodePtr stream_switch_node, std::set<NodePtr> &group_nodes,
                             std::vector<NodePtr> &merge_nodes, std::set<std::string> &stream_labels);

  Status MappingNodeToSwitchAndMerge(const std::set<NodePtr> &group_set, const std::vector<NodePtr> &merge_vec,
                                     const NodePtr &cast_node, const NodePtr &switch_node,
                                     std::map<NodePtr, std::pair<std::set<NodePtr>, NodePtr>> &node_2_switch_merge);

  bool IsBigSmallLoopStreamSwitch(OpDescPtr switch_op_desc);
  bool IsWhileStreamSwitch(OpDescPtr switch_op_desc);
  bool IsIndirectConnect(const NodePtr &node_a, const NodePtr &node_b);
};
}  // namespace ge
#endif // GE_GRAPH_PASSES_PARALLEL_GROUP_PASS_H
