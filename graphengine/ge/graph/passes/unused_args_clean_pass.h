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
#ifndef GE_COMMON_CASE_ARGS_CLEAN_H_
#define GE_COMMON_CASE_ARGS_CLEAN_H_

#include "external/graph/types.h"
#include "inc/graph_pass.h"

#include <map>
#include <set>
#include <vector>
#include <string>

using std::set;
using std::map;

namespace ge {
class UnusedArgsCleanPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph) override;

 private:
  ///
  /// @ingroup ge
  /// @brief Create nodes for root graph.
  /// @param [in] graph_nodes: Data groups of subgraph.
  /// @param [in] func_node: functional Node of Case.
  /// @param [in] parent_index: parent index for check.
  /// @return true: unused / false: used
  ///
  bool UnusedInputTensor(const map<ComputeGraphPtr, map<uint32_t, NodePtr>> &graph_nodes,
                         const NodePtr &func_node, uint32_t parent_index);

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
  /// @brief Remove Case input Tensor.
  /// @param [in] graph_nodes: Data groups of subgraph.
  /// @param [in] func_node: functional Node of Case.
  /// @param [in] parent_index: parent index for remove.
  /// @return 0: SUCCESS / others: FAILED
  ///
  Status RemoveInputTensor(const map<ComputeGraphPtr, map<uint32_t, NodePtr>> &graph_nodes,
                           const NodePtr &func_node, uint32_t parent_index);

  ///
  /// @ingroup ge
  /// @brief Update Case input Tensor.
  /// @param [in] graph_nodes: Data groups of subgraph.
  /// @param [in] func_node: functional Node of Case.
  /// @param [in] parent_index: parent index for update.
  /// @param [in] unused_num: unused args num.
  /// @return 0: SUCCESS / others: FAILED
  ///
  Status UpdateInputTensor(const map<ComputeGraphPtr, map<uint32_t, NodePtr>> &graph_nodes,
                           const NodePtr &func_node, uint32_t parent_index, uint32_t unused_num);
};
}  // namespace ge
#endif  // GE_COMMON_CASE_ARGS_CLEAN_H_
