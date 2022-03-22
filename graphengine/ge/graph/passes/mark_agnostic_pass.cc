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
#include "graph/passes/mark_agnostic_pass.h"

#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
const size_t kTwoInputNodesSize = 2;

Status MarkAgnosticPass::Run(ComputeGraphPtr graph) {
  for (const auto &node : graph->GetDirectNode()) {
    auto node_type = NodeUtils::GetNodeType(*node);
    if (node_type == SWITCH || node_type == SWITCHN) {
      GELOGD("Mark format agnostic and continuous for switch node %s", node->GetName().c_str());
      const OpDescPtr op_desc = node->GetOpDesc();
      const GeTensorDescPtr op_tensor = op_desc->MutableInputDesc(0);
      if (op_tensor == nullptr) {
        GELOGD("Op: %s, Index:0,has no input", node->GetName().c_str());
        continue;
      }
      AttrUtils::SetInt(op_tensor, ATTR_NAME_FORMAT_CONTINUOUS, 1);
      AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_FORMAT_AGNOSTIC, 1);
      AttrUtils::SetListInt(node->GetOpDesc(), ATTR_NAME_FORMAT_AGNOSTIC_EXCEPT_INPUT, std::vector<int64_t>({1}));
      continue;
    }
    if (node_type == IDENTITY) {
      GELOGD("Mark format agnostic for identity node %s", node->GetName().c_str());
      AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_FORMAT_AGNOSTIC, 1);
      continue;
    }
    if (node_type == REFMERGE || node_type == REFSWITCH) {
      GELOGD("Mark format agnostic for regmerge and refswitch node %s", node->GetName().c_str());
      AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_FORMAT_AGNOSTIC, 1);
      AttrUtils::SetListInt(node->GetOpDesc(), ATTR_NAME_FORMAT_AGNOSTIC_EXCEPT_INPUT, std::vector<int64_t>({1}));
      continue;
    }
    if (node_type == MERGE) {
      GELOGD("Mark format agnostic and continuous for merge node %s", node->GetName().c_str());

      // Always set continuous attr for merge output 0
      GE_CHK_STATUS_RET(SetContinuousAttr(node, {0}));

      // Merge-->NetOutput only set merge output 0's continuous attr
      const auto &output_nodes = node->GetOutDataNodes();
      if (!output_nodes.empty()) {
        if (output_nodes.at(0)->GetType() == NETOUTPUT) {
          continue;
        }
      }
      // Set format agnostic attr for merge in and out tensordesc
      AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_FORMAT_AGNOSTIC, 1);
      AttrUtils::SetListInt(node->GetOpDesc(), ATTR_NAME_FORMAT_AGNOSTIC_EXCEPT_OUTPUT, std::vector<int64_t>({1}));

      // Set attr for enter and nextiteration
      if (HandWhileLoop(node) != SUCCESS) {
        GELOGE(FAILED, "[Hand][WhileLoop] for node:%s failed.", node->GetName().c_str());
        return FAILED;
      }
      continue;
    }
  }

  return SUCCESS;
}

bool MarkAgnosticPass::IsWhileLoop(const NodePtr &merge_node, NodePtr &enter, NodePtr &next) {
  auto node_type = NodeUtils::GetNodeType(*merge_node);
  if (node_type != MERGE) {
    GELOGW("Node %s type %s is not merge op.", merge_node->GetName().c_str(), node_type.c_str());
    return false;
  }
  /// Enter-----------+
  ///                 +-> Merge
  /// NextIteration---+
  auto input_nodes = merge_node->GetInDataNodes();
  if (input_nodes.size() != kTwoInputNodesSize) {
    GELOGD("Node %s type %s with [data input size[%zu]] is not enter-merge-nextiteration target.",
           merge_node->GetName().c_str(), node_type.c_str(), input_nodes.size());
    return false;
  }
  auto in_node0 = input_nodes.at(0);
  auto in_node1 = input_nodes.at(1);
  auto in_type0 = NodeUtils::GetNodeType(in_node0);
  auto in_type1 = NodeUtils::GetNodeType(in_node1);
  if ((in_type0 != ENTER || in_type1 != NEXTITERATION) && (in_type0 != NEXTITERATION || in_type1 != ENTER)) {
    GELOGD("Node %s type %s with [data input0's type %s input1's type %s] is not enter-merge-nextiteration target.",
           merge_node->GetName().c_str(), node_type.c_str(), in_type0.c_str(), in_type1.c_str());
    return false;
  }
  enter = in_node0;
  next = in_node1;
  return true;
}

Status MarkAgnosticPass::HandWhileLoop(const NodePtr &node) {
  NodePtr enter = nullptr;
  NodePtr next = nullptr;
  if (!IsWhileLoop(node, enter, next)) {
    return SUCCESS;
  }
  GE_CHECK_NOTNULL(enter);
  GE_CHECK_NOTNULL(next);
  // Set continuous attr
  GE_CHK_STATUS_RET(SetContinuousAttr(enter, {0}));
  GE_CHK_STATUS_RET(SetContinuousAttr(next, {0}));
  // Set format agnostic attr
  (void)AttrUtils::SetInt(enter->GetOpDesc(), ATTR_NAME_FORMAT_AGNOSTIC, 1);
  (void)AttrUtils::SetInt(next->GetOpDesc(), ATTR_NAME_FORMAT_AGNOSTIC, 1);

  return SUCCESS;
}

Status MarkAgnosticPass::SetContinuousAttr(const NodePtr &node, const std::vector<uint32_t> &indexes) {
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  // This flag is for fe performance optimization
  (void)AttrUtils::SetBool(op_desc, ATTR_NAME_REFRESH_CONTINUOUS_FLAG, true);
  for (auto index : indexes) {
    auto out = op_desc->MutableOutputDesc(index);
    if (out == nullptr) {
      REPORT_INNER_ERROR("E19999", "Op:%s(%s) output:%u desc is nullptr, check invalid",
                         op_desc->GetName().c_str(), op_desc->GetType().c_str(), index);
      GELOGE(FAILED, "[Check][Param]Op:%s(%s) output:%u desc is nullptr",
             op_desc->GetName().c_str(), op_desc->GetType().c_str(), index);
      return FAILED;
    }
    // This attr is for out's dtype and format continuous with it's peer input
    (void)AttrUtils::SetInt(out, ATTR_NAME_FORMAT_CONTINUOUS, 1);
  }

  return SUCCESS;
}
}  // namespace ge