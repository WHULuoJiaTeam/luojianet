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

#ifndef GE_GRAPH_PASSES_CAST_REMOVE_PASS_H_
#define GE_GRAPH_PASSES_CAST_REMOVE_PASS_H_

#include <vector>
#include "graph/passes/base_pass.h"

namespace ge {
class CastRemovePass : public BaseNodePass {
 public:
  Status Run(NodePtr &node) override;

 private:
  bool CheckPrecisionLoss(const std::vector<NodePtr> &nodes_to_fuse);
  bool HasSameDataType(OpDescPtr &begin_op_desc, OpDescPtr &end_op_desc, DataType &type) const;
  Status RemoveCast(DataType &type, std::vector<NodePtr> &nodes_to_fuse);
  NodePtr GetTheEndNode(NodePtr begin_node, std::vector<NodePtr> &nodes_to_fuse);
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_CAST_REMOVE_PASS_H_
