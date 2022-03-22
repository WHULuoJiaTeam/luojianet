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
#include "graph/passes/hccl_tailing_optimization_pass.h"
#include "common/transop_util.h"

namespace ge {
Status HcclTailingOptimizationPass::Run(ComputeGraphPtr graph) {
  for (const auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (node->GetType() != HCOMALLREDUCE) {
      continue;
    }
    for (auto &out_node : node->GetOutDataNodes()) {
      if (!TransOpUtil::IsTransOp(out_node)) {
        continue;
      }

      GE_CHK_STATUS_RET_NOLOG(CopyControlEdgesForTransOp(out_node));
    }
  }
  return SUCCESS;
}
Status HcclTailingOptimizationPass::CopyControlEdgesForTransOp(NodePtr &first_trans_op) {
  auto dst_in_ctrl_anchor = first_trans_op->GetInControlAnchor();
  GE_CHECK_NOTNULL(dst_in_ctrl_anchor);
  std::set<OutControlAnchorPtr> src_out_ctrl_anchors;
  std::vector<NodePtr> trans_op_nodes{first_trans_op};

  while (!trans_op_nodes.empty()) {
    auto trans_op_node = trans_op_nodes.back();
    trans_op_nodes.pop_back();

    for (auto &next_node : trans_op_node->GetOutDataNodes()) {
      auto in_ctrl_anchor = next_node->GetInControlAnchor();
      GE_CHECK_NOTNULL(in_ctrl_anchor);

      auto peer_out_ctrl_anchors = in_ctrl_anchor->GetPeerOutControlAnchors();

      for (auto src_ctrl_anchor : peer_out_ctrl_anchors) {
        GE_CHECK_NOTNULL(src_ctrl_anchor->GetOwnerNode());
        src_out_ctrl_anchors.emplace(src_ctrl_anchor);
      }
      if (TransOpUtil::IsTransOp(next_node)) {
        trans_op_nodes.emplace_back(next_node);
      }
    }
  }

  for (auto &src_out_ctrl_anchor : src_out_ctrl_anchors) {
    if (!src_out_ctrl_anchor->IsLinkedWith(dst_in_ctrl_anchor)) {
      GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(src_out_ctrl_anchor, dst_in_ctrl_anchor),
                              "[Add][Edge] between %s->%s failed",
                              src_out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                              first_trans_op->GetName().c_str());
    }
  }

  return SUCCESS;
}
}  // namespace ge
