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

#include "graph/passes/mark_same_addr_pass.h"

namespace ge {
bool MarkSameAddrPass::IsNextNodeExpected(const ge::NodePtr &cur_node, const vector<string> &next_nodes,
                                          int &out_anchor_idx) {
  for (auto out_anchor : cur_node->GetAllOutDataAnchors()) {
    if (out_anchor == nullptr) {
      continue;
    }
    for (auto in_anchor : out_anchor->GetPeerInDataAnchors()) {
      if (in_anchor == nullptr) {
        continue;
      }
      auto dst_node = in_anchor->GetOwnerNode();
      if (dst_node == nullptr) {
        continue;
      }
      if (std::count(next_nodes.begin(), next_nodes.end(), dst_node->GetType()) > 0) {
        out_anchor_idx = out_anchor->GetIdx();
        GELOGD("Current node is %s, next node is %s.", cur_node->GetName().c_str(), dst_node->GetName().c_str());
        return true;
      }
    }
  }
  return false;
}

Status MarkSameAddrPass::Run(ComputeGraphPtr graph) {
  GELOGD("MarkSameAddrPass begin.");
  GE_CHECK_NOTNULL(graph);
  if (graph->GetGraphUnknownFlag()) {
    GELOGD("Graph[%s] is unknown shape, do not need to set fixed addr attr.", graph->GetName().c_str());
    return SUCCESS;
  }

  int out_anchor_idx = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    vector<string> next_nodes = {STREAMSWITCH, STREAMSWITCHN, LABELSWITCHBYINDEX};
    if (IsNextNodeExpected(node, next_nodes, out_anchor_idx)) {
      string tensor_name = op_desc->GetOutputNameByIndex(out_anchor_idx);
      (void)ge::AttrUtils::SetStr(node->GetOpDesc(), ATTR_DYNAMIC_SHAPE_FIXED_ADDR, tensor_name);
      (void)ge::AttrUtils::SetInt(node->GetOpDesc(), ATTR_DYNAMIC_SHAPE_FIXED_ADDR_INDEX, out_anchor_idx);
    }
  }
  GELOGD("MarkSameAddrPass end.");
  return SUCCESS;
}
}  // namespace ge
