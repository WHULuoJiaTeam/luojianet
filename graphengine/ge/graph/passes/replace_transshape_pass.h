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

#ifndef GE_GRAPH_PASSES_REPLACE_TRANS_SHAPE_PASS_H_
#define GE_GRAPH_PASSES_REPLACE_TRANS_SHAPE_PASS_H_

#include "inc/graph_pass.h"

namespace ge {
class ReplaceTransShapePass : public GraphPass {
 public:
  Status Run(ge::ComputeGraphPtr graph) override;

 private:
  Status ReplaceTransShapeNode(ComputeGraphPtr &graph, NodePtr &trans_shape_node);
  void CopyControlEdges(NodePtr &old_node, NodePtr &new_node, bool input_check_flag = false);
  void RemoveControlEdges(NodePtr &node);
  void ReplaceControlEdges(NodePtr &old_node, NodePtr &new_node);
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_REPLACE_TRANS_SHAPE_PASS_H_
