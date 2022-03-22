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

#ifndef GE_GRAPH_PASSES_VAR_IS_INITIALIZED_OP_PASS_H_
#define GE_GRAPH_PASSES_VAR_IS_INITIALIZED_OP_PASS_H_
#include <map>
#include <memory>
#include <set>
#include <vector>
#include "graph/passes/base_pass.h"

namespace ge {
class VarIsInitializedOpPass : public BaseNodePass {
 public:
  Status Run(NodePtr &node) override;

 private:
  Status CheckSrcNode(const NodePtr &node, bool &inited) const;
  Status CreateConstant(NodePtr &node, OpDescPtr &op_desc, bool inited);
  Status ProcessInAnchor(NodePtr &node, NodePtr &new_node);
  Status ChangeNodeToConstant(NodePtr &node, bool inited);
  Status UpdateInitedVars(const NodePtr &node);
  Status CheckAndSetVarInited(const NodePtr &node, bool &inited, int64_t &inited_var);
  std::set<int64_t> *CreateInitedVars();
  bool IsVarInitedOnTheGraphAndNode(const NodePtr &node, int64_t var_id) const;

  std::vector<std::unique_ptr<std::set<int64_t>>> var_inited_keeper_;
  std::map<int64_t, std::set<int64_t> *> nodes_to_inited_vars_;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_VAR_IS_INITIALIZED_OP_PASS_H_
