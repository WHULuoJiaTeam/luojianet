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

#ifndef GE_GRAPH_PASSES_REF_IDENTITY_DELETE_OP_PASS_H_
#define GE_GRAPH_PASSES_REF_IDENTITY_DELETE_OP_PASS_H_

#include <map>
#include <string>
#include "framework/common/ge_inner_error_codes.h"
#include "inc/graph_pass.h"

namespace ge {
class RefIdentityDeleteOpPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph);

 private:
  Status DealNoOutputRef(const NodePtr &node, const NodePtr &ref_identity, int input_index,
                         const ComputeGraphPtr &graph);
  NodePtr GetVariableRef(const NodePtr &ref, const NodePtr &ref_identity, NodePtr &first_node);
  bool CheckControlEdge(const NodePtr &ref, const NodePtr &variable_ref);
  Status RemoveUselessControlEdge(const NodePtr &ref, const NodePtr &variable_ref);
  NodePtr GetRefNode(const NodePtr &node, int &input_index);
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_REF_IDENTITY_DELETE_OP_PASS_H_
