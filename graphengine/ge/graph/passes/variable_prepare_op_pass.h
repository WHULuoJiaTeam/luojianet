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

#ifndef GE_GRAPH_PASSES_VARIABLE_PREPARE_OP_PASS_H_
#define GE_GRAPH_PASSES_VARIABLE_PREPARE_OP_PASS_H_

#include <map>
#include <stack>
#include <string>

#include "framework/common/ge_inner_error_codes.h"
#include "inc/graph_pass.h"

namespace ge {
class VariablePrepareOpPass : public GraphPass {
 public:
  Status Run(ge::ComputeGraphPtr graph);

 private:
  Status DealVariableNode(ge::NodePtr &node);
  Status DealWritableNode(const ge::NodePtr &writable_node, int input_index, int output_index,
                          const ge::NodePtr &var_node);
  Status GetPeerNodeOfRefOutput(const ge::NodePtr &node, int output_index,
                                std::stack<pair<NodePtr, pair<int, int>>> &nodes);
  Status AddVariableRef(ge::NodePtr &node, const ge::NodePtr &var_node, int index);
  Status InsertVariableRef(ge::NodePtr &node, int in_index, const ge::NodePtr &var_node);
  Status AddControlEdge(const ge::NodePtr &node, const ge::NodePtr &variable_ref_node);
  NodePtr CreateVariableRef(const std::string &variable_ref_name, const ge::NodePtr &var_node);
  NodePtr CreateRefIdentity(const std::string &ref_identity_name, const ge::NodePtr &node, uint32_t input_index);
  void GetWritableNodeOutIndex(const NodePtr &node, int input_index, std::vector<int> &output_indexes);
  void GenerateRefTypeAndInputOutputMap(const NodePtr &node);
  void FindRefOutIndex(const std::string &node_type, int input_index,
                       const std::map<std::string, std::map<int, vector<int>>> &ref_map,
                       std::vector<int> &output_indexes);
  Status CheckStreamLabel(const ge::NodePtr &var_ref_node, const ge::NodePtr &final_writable_node);
  bool HasControlOut(const ge::NodePtr &node);

  std::map<std::string, std::map<int, std::vector<int>>> ref_input_output_map_;
  static std::map<std::string, std::map<int, std::vector<int>>> ref_node_without_prototype_map_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_VARIABLE_PREPARE_OP_PASS_H_
