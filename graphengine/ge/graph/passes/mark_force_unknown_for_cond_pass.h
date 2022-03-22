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

#ifndef GE_GRAPH_PASSES_MARK_FORCE_UNKNOWN_FOR_COND_PASS_H_
#define GE_GRAPH_PASSES_MARK_FORCE_UNKNOWN_FOR_COND_PASS_H_

#include "inc/graph_pass.h"

#include <queue>

namespace ge {
class MarkForceUnknownForCondPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph);

 private:
  ///
  /// @brief Deal with Switch node for LoopCond
  /// @param [in] Switch node
  /// @param [in] dest span
  /// @param [out] Search queue
  /// @return true: Switch In while loop / false: Not in while Loop.
  ///
  bool DealAsLoopSwitch(const NodePtr &node, uint32_t dst_span, std::queue<std::pair<NodePtr, uint32_t>> &search_queue);

  ///
  /// @brief Mark force unknown shape for Switch node
  /// @param [in] merge node
  /// @param [out] switch group
  /// @return
  ///
  void MarkUnknownForSwitch(const NodePtr &node, std::vector<NodePtr> &switch_group);

  ///
  /// @brief Mark force unknown shape for Switch node
  /// @param [in] switch groups
  /// @return
  ///
  void MarkUnknownForSwitch(const std::map<NodePtr, std::vector<NodePtr>> &switch_groups);
};
} // namespace ge
#endif  // GE_GRAPH_PASSES_MARK_FORCE_UNKNOWN_FOR_COND_PASS_H_
