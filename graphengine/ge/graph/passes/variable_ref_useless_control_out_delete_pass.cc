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
#include "graph/passes/variable_ref_useless_control_out_delete_pass.h"

namespace ge {
Status VariableRefUselessControlOutDeletePass::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() != VARIABLE) {
      continue;
    }
    std::string src_var_name;
    if (!AttrUtils::GetStr(node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, src_var_name)) {
      continue;
    }
    auto src_nodes = node->GetInDataNodes();
    if (src_nodes.empty()) {
      GELOGW("The variable ref name %s(ref %s) does not has a input node",
             node->GetName().c_str(), src_var_name.c_str());
      continue;
    }
    auto &src_node = src_nodes.at(0);
    auto controlled_nodes_vec = src_node->GetOutNodes();
    std::set<NodePtr> controlled_nodes{controlled_nodes_vec.begin(), controlled_nodes_vec.end()};

    auto out_control_anchor = node->GetOutControlAnchor();
    for (const auto &dst_node_anchor : out_control_anchor->GetPeerInControlAnchors()) {
      if (controlled_nodes.count(dst_node_anchor->GetOwnerNode()) > 0) {
        GELOGI("Unlink the duplicated control edge from variable ref %s to %s, prev node %s",
               node->GetName().c_str(),
               dst_node_anchor->GetOwnerNode()->GetName().c_str(),
               src_node->GetName().c_str());
        out_control_anchor->Unlink(dst_node_anchor);
      }
    }
  }
  return SUCCESS;
}
}