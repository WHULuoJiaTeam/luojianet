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

#include "graph/passes/mark_force_unknown_for_cond_pass.h"

#include "graph/utils/node_utils.h"
#include "common/omg_util.h"

namespace ge {
namespace {
inline bool IsMergeInLoop(const NodePtr &node) {
  const static std::set<std::string> kLoopMergeInputs{ ENTER, REFENTER, NEXTITERATION, REFNEXTITERATION };

  return kLoopMergeInputs.count(NodeUtils::GetNodeType(node)) > 0;
}
}

Status MarkForceUnknownForCondPass::Run(ComputeGraphPtr graph) {
  GELOGD("MarkForceUnknownForCondPass Enter");
  std::map<NodePtr, std::vector<NodePtr>> switch_groups;
  for (const auto &node : graph->GetDirectNode()) {
    if (kMergeOpTypes.count(NodeUtils::GetNodeType(node)) == 0) {
      continue;
    }

    const auto &all_in_nodes = node->GetInDataNodes();
    if (std::any_of(all_in_nodes.begin(), all_in_nodes.end(), IsMergeInLoop)) {
      continue;  // LoopCond marked in NextIterationPass.
    }

    MarkUnknownForSwitch(node, switch_groups[node]);
  }

  MarkUnknownForSwitch(switch_groups);
  GELOGD("MarkForceUnknownForCondPass Leave");
  return SUCCESS;
}

///
/// @brief Deal with Switch node for LoopCond
/// @param [in] Switch node
/// @param [in] dest span
/// @param [out] Search queue
/// @return true: Switch In while loop / false: Not in while Loop.
///
bool MarkForceUnknownForCondPass::DealAsLoopSwitch(const NodePtr &node, uint32_t dst_span,
                                                   std::queue<std::pair<NodePtr, uint32_t>> &search_queue) {
  ///                 LoopCond --->\.
  ///                               \.
  /// Enter-----------+              \.
  ///                 +--> Merge --> Switch --> Exit
  /// NextIteration---+
  const auto is_loop_op = [](const NodePtr &n) {
    return NodeUtils::GetNodeType(n) == LOOPCOND;
  };
  const auto is_exit_op = [](const NodePtr &n) {
    return kExitOpTypes.count(NodeUtils::GetNodeType(n)) > 0;
  };

  const auto src_nodes = node->GetInAllNodes();
  const auto dst_nodes = node->GetOutAllNodes();
  if (std::none_of(src_nodes.begin(), src_nodes.end(), is_loop_op) &&
      std::none_of(dst_nodes.begin(), dst_nodes.end(), is_exit_op)) {
    return false;
  }

  for (const auto &m : src_nodes) {
    if (kMergeOpTypes.count(NodeUtils::GetNodeType(m)) > 0) {
      for (const auto &n : m->GetInAllNodes()) {
        if (kNextIterationOpTypes.count(NodeUtils::GetNodeType(n)) > 0) {
          continue;
        }

        search_queue.push({n, dst_span});
        GELOGD("Travel in Loop: %s <-- %s <-- %s, span is: %u", node->GetName().c_str(), m->GetName().c_str(),
               n->GetName().c_str(), dst_span);
      }
    }
  }

  return true;
}

///
/// @brief Mark force unknown shape for Switch node
/// @param [in] merge node
/// @param [out] switch group
/// @return
///
void MarkForceUnknownForCondPass::MarkUnknownForSwitch(const NodePtr &node, std::vector<NodePtr> &switch_group) {
  // Switch --> {Switch --> Merge} --> Merge
  GELOGD("Search Switch node for Merge: %s", node->GetName().c_str());
  std::unordered_set<NodePtr> nodes_seen;
  std::queue<std::pair<NodePtr, uint32_t>> search_queue({{node, 0}});
  while (!search_queue.empty()) {
    const auto dst_node = search_queue.front().first;
    const auto dst_span = search_queue.front().second;
    search_queue.pop();

    for (const auto &in_node : dst_node->GetInAllNodes()) {
      if (nodes_seen.count(in_node) > 0) {
        GELOGD("Travel node: %s, Skip already seen node: %s", dst_node->GetName().c_str(), in_node->GetName().c_str());
        continue;
      }
      nodes_seen.insert(in_node);

      const std::string node_type = NodeUtils::GetNodeType(in_node);
      GELOGD("Travel node: %s, %s node: %s, span is: %u", dst_node->GetName().c_str(), node_type.c_str(),
             in_node->GetName().c_str(), dst_span);
      if (kSwitchOpTypes.count(node_type) > 0) { // Switch input node.
        if (DealAsLoopSwitch(in_node, dst_span, search_queue)) {
          continue;
        }

        if (dst_span > 0) {
          search_queue.push({in_node, dst_span - 1});
        } else {
          switch_group.emplace_back(in_node);
        }
      } else if (kMergeOpTypes.count(node_type) > 0) { // Merge input node.
        search_queue.push({in_node, dst_span + 1});
      } else {
        search_queue.push({in_node, dst_span});
      }
    }
  }
}

///
/// @brief Mark force unknown shape for Switch node
/// @param [in] switch groups
/// @return
///
void MarkForceUnknownForCondPass::MarkUnknownForSwitch(const std::map<NodePtr, std::vector<NodePtr>> &switch_groups) {
  // Step 0: no group assigned. such as:
  // Merge1{id=0, group=} => {Switch1{id=1, group=}, Switch2{id=2, group=}}
  // Merge2{id=3, group=} => {Switch1{id=1, group=}, Switch3{id=4, group=}}
  // Merge3{id=5, group=} => {Switch4{id=6, group=}, Switch5{id=7, group=}}
  // Merge4{id=8, group=} => {Switch1{id=1, group=}, Switch5{id=7, group=}}
  std::map<int64_t, int64_t> unique_groups;
  const auto get_group_index = [&unique_groups](const NodePtr &merge, const std::vector<NodePtr> &switch_group) {
    int64_t group_index = merge->GetOpDesc()->GetId();
    std::set<int64_t> group_ids{group_index};
    for (const auto &node : switch_group) {
      if (AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_CONTROL_FLOW_GROUP, group_index)) {
        GELOGI("[%s] Get group from [%s], index[%ld]", merge->GetName().c_str(), node->GetName().c_str(), group_index);
        group_ids.insert(group_index);
      }
    }

    const auto it = unique_groups.find(group_index);
    if (it != unique_groups.end()) {
      group_index = it->second;
    }

    for (auto id : group_ids) {
      unique_groups[id] = group_index;
    }

    return group_index;
  };

  const auto set_group_index = [](const NodePtr &merge, const std::vector<NodePtr> &switch_group, int64_t group_index) {
    SetControlFlowGroup(merge, group_index);
    for (const auto &node : switch_group) {
      SetControlFlowGroup(node, group_index);
    }
  };

  // Step 1: Set group index to merge, if switch already has group, use assigned group.
  // Merge1{id=0, group=0} => {Switch1{id=1, group=0}, Switch2{id=2, group=0}}
  // Merge2{id=3, group=0} => {Switch1{id=1, group=0}, Switch3{id=4, group=0}}
  // Merge3{id=5, group=5} => {Switch4{id=6, group=5}, Switch5{id=7, group=5}}
  // Merge4{id=8, group=0} => {Switch1{id=1, group=0}, Switch5{id=7, group=0}}
  for (const auto group : switch_groups) {
    int64_t group_index = get_group_index(group.first, group.second);
    set_group_index(group.first, group.second, group_index);
  }

  // Step 2: Adjust crossed merge group for unique group.
  // Merge1{id=0, group=0} => {Switch1{id=1, group=0}, Switch2{id=2, group=0}}
  // Merge2{id=3, group=0} => {Switch1{id=1, group=0}, Switch3{id=4, group=0}}
  // Merge3{id=5, group=0} => {Switch4{id=6, group=0}, Switch5{id=7, group=0}}
  // Merge4{id=8, group=0} => {Switch1{id=1, group=0}, Switch5{id=7, group=0}}
  for (const auto group : switch_groups) {
    int64_t group_index = -1;
    (void)AttrUtils::GetInt(group.first->GetOpDesc(), ATTR_NAME_CONTROL_FLOW_GROUP, group_index);

    const auto it = unique_groups.find(group_index);
    if (it != unique_groups.end() && it->first != it->second) {
      set_group_index(group.first, group.second, it->second);
    }
  }
}
} // namespace ge
