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

#ifndef GE_GRAPH_PASSES_GLOBAL_STEP_INSERT_PASS_H_
#define GE_GRAPH_PASSES_GLOBAL_STEP_INSERT_PASS_H_

#include <string>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "inc/graph_pass.h"

namespace ge {
///
/// Add global step op to the computeGraph when needed.
/// [Notice]: this pass must work before graph partitioner start work
///  in order to make the global step variable place in known subgraph
///
class GlobalStepInsertPass : public GraphPass {
public:
  ///
  /// @param compute_graph graph
  /// @return SUCCESS: do success
  ///         NOT_CHANGED : do nothing
  ///         Other: failed
  ///
  Status Run(ComputeGraphPtr compute_graph) override;
private:
  ///
  /// Universal insert node to graph.
  /// @param compute_graph graph
  /// @param node_type inserted node type
  /// @param node_name inserted node name
  /// @param input_list input desc list
  /// @param output_list output desc list
  /// @return the inserted node. if insert failed return nullptr.
  ///
  NodePtr InsertOp(ComputeGraphPtr &compute_graph,
                   const string &node_type,
                   const string &node_name,
                   const std::vector<GeTensorDesc> &input_list,
                   const std::vector<GeTensorDesc> &output_list);
};
} // namespace ge

#endif  // GE_GRAPH_PASSES_GLOBAL_STEP_INSERT_PASS_H_