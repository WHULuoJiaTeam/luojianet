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

#ifndef GE_GRAPH_PASSES_VARIABLE_FORMAT_PASS_H_
#define GE_GRAPH_PASSES_VARIABLE_FORMAT_PASS_H_

#include <map>
#include <set>
#include <string>
#include "graph/types.h"
#include "graph/utils/op_desc_utils.h"
#include "inc/graph_pass.h"

namespace ge {
class VariableFormatPass : public GraphPass {
 public:
  Status Run(ge::ComputeGraphPtr graph) override;

 private:
  bool GetApplyMomentumOpByVariableInput(const ge::NodePtr &var_node, ge::NodePtr &use_node);

  bool ConfirmUseOpAndIndexByAnchor(const ge::InDataAnchorPtr &in_anchor,
                                    const map<string, std::set<int> > &confirm_ops, ge::NodePtr &use_node);

  Status UpdateApplyMomentumInputFormat(const ge::NodePtr &node);

  Status UpdateVariableOutFormat(const ge::NodePtr &var_node, ge::NodePtr &use_node);
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_VARIABLE_FORMAT_PASS_H_
