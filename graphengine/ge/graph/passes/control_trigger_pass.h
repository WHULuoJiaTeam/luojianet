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

#ifndef GE_GRAPH_PASSES_CONTROL_TRIGGER_PASS_H_
#define GE_GRAPH_PASSES_CONTROL_TRIGGER_PASS_H_

#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include "inc/graph_pass.h"

namespace ge {
enum ControlNodeType {
  kNotControlOp,
  kCondSwitch,
  kCondMerge,
  kLoopSwitchT,
  kLoopSwitchF,
  kEnter,
  kInvalidType
};

class ControlTriggerPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph);
  Status ClearStatus() override;

 private:
  Status HandleDynamicCtrlEdges(ComputeGraphPtr &graph, NodePtr &node, NodePtr &in_ctrl_node);
  Status FindSwitchNode(const NodePtr &node, NodePtr &switch_node, bool &branch_flag);
  ControlNodeType TransferNodeType(const NodePtr &node, uint32_t index);
  void GetInNodes(const NodePtr &node, std::set<std::pair<NodePtr, uint32_t>> &in_nodes);
  Status InsertOppositeBranch(ComputeGraphPtr &graph, NodePtr &node, NodePtr &in_ctrl_node, NodePtr &switch_node,
                              bool branch_flag);
  NodePtr InsertMergeNode(ComputeGraphPtr &graph, NodePtr &node, NodePtr &in_ctrl_node, const GeTensorDesc &data_desc);
  NodePtr InsertConstNode(ComputeGraphPtr &graph, NodePtr &merge_node, const GeTensorDesc &data_desc, bool flag);
  NodePtr InsertIdentityNode(ComputeGraphPtr &graph, const std::string &name, const GeTensorDesc &data_desc);
  Status FindPredInput(const NodePtr &switch_node);

  // <switch_node, pred_node>
  std::unordered_map<NodePtr, NodePtr> switch_cond_map_;
  // <ControlTrigger, <pred_node, {const_f, const_t}>>
  std::unordered_map<NodePtr, std::unordered_map<NodePtr, std::pair<NodePtr, NodePtr>>> control_trigger_map_;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_CONTROL_TRIGGER_PASS_H_
