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

#include "graph/passes/enter_pass.h"

#include "graph/debug/ge_attr_define.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "graph/utils/graph_utils.h"

namespace {
const size_t kOutNodesNum = 1;
const size_t kInCtrlNodesNum = 1;
}

namespace ge {
Status EnterPass::Run(NodePtr &node) {
  GELOGD("EnterPass running");
  GE_CHECK_NOTNULL(node);

  if ((node->GetType() != ENTER) && (node->GetType() != REFENTER)) {
    return SUCCESS;
  }

  // enter node has only one input
  if (node->GetInDataNodes().empty()) {
    REPORT_INNER_ERROR("E19999", "Param node in data nodes is empty, check invalid");
    GELOGE(PARAM_INVALID, "[Check][Param] enter_node %s has no input", node->GetName().c_str());
    return PARAM_INVALID;
  }
  NodePtr in_node = node->GetInDataNodes().at(0);
  GE_CHECK_NOTNULL(in_node);

  if ((in_node->GetType() != CONSTANT) && (in_node->GetType() != CONSTANTOP)) {
    return SUCCESS;
  }

  bool need_remove_flag = in_node->GetInControlNodes().empty() && node->GetInControlNodes().empty();
  if (!need_remove_flag) {
    return SUCCESS;
  }
  if (node->GetOutDataNodes().empty()) {
    for (auto &out_ctrl_node : node->GetOutControlNodes()) {
      if (out_ctrl_node == nullptr) {
        continue;
      }
      GELOGI("Remove control edge from %s to %s.", node->GetName().c_str(), out_ctrl_node->GetName().c_str());
      if (GraphUtils::RemoveEdge(node->GetOutControlAnchor(), out_ctrl_node->GetInControlAnchor()) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Remove control edge between op:%s(%s) and op:%s(%s) failed",
                          node->GetName().c_str(), node->GetType().c_str(),
                          out_ctrl_node->GetName().c_str(), out_ctrl_node->GetType().c_str());
        GELOGE(FAILED, "[Remove][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
               node->GetName().c_str(), node->GetType().c_str(),
               out_ctrl_node->GetName().c_str(), out_ctrl_node->GetType().c_str());
        return FAILED;
      }
    }
  } else {
    if (OptimizeEnterWithOnlyDataOut(node, in_node) != SUCCESS) {
      GELOGE(FAILED, "[Optimize][EnterNode] [%s] with only out data node failed.", node->GetName().c_str());
      return FAILED;
    }
    if (UnlinkCtrlEdgeBeforeConst(node) != SUCCESS) {
      GELOGE(FAILED, "[Unlink][ControlEdge] before const of node[%s]'s out nodes failed.", node->GetName().c_str());
      return FAILED;
    }
  }

  GELOGD("EnterPass success");
  return SUCCESS;
}

Status EnterPass::OptimizeEnterWithOnlyDataOut(NodePtr &node, NodePtr &in_node) {
  if ((in_node->GetOutAllNodes().size() != kOutNodesNum) || !node->GetOutControlNodes().empty()) {
    return SUCCESS;
  }
  bool is_constant_flag = true;
  (void)AttrUtils::GetBool(node->GetOpDesc(), ENTER_ATTR_CONSTANT_FLAG, is_constant_flag);
  if (!is_constant_flag) {
    return SUCCESS;
  }

  GE_CHECK_NOTNULL(in_node->GetOutDataAnchor(0));
  GE_CHK_GRAPH_STATUS_RET(in_node->GetOutDataAnchor(0)->Unlink(node->GetInDataAnchor(0)))
  const auto &out_data_anchor = node->GetOutDataAnchor(0);
  GE_CHECK_NOTNULL(out_data_anchor);
  for (const auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    GE_CHK_GRAPH_STATUS_RET(out_data_anchor->Unlink(peer_in_data_anchor))
    GE_CHK_GRAPH_STATUS_RET(in_node->GetOutDataAnchor(0)->LinkTo(peer_in_data_anchor))
  }
  GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveNodeWithoutRelink(node->GetOwnerComputeGraph(), node))
  AddNodeDeleted(node);
  AddRePassNodesWithInOut(in_node);

  return SUCCESS;
}

Status EnterPass::UnlinkCtrlEdgeBeforeConst(NodePtr &node) {
  auto out_ctrl_nodes = node->GetOutControlNodes();
  if (out_ctrl_nodes.empty()) {
    return SUCCESS;
  }
  auto out_ctrl_anchor = node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(out_ctrl_anchor);

  for (auto &out_ctrl_node : out_ctrl_nodes) {
    GE_CHECK_NOTNULL(out_ctrl_node);
    if ((out_ctrl_node->GetType() != CONSTANT) && (out_ctrl_node->GetType() != CONSTANTOP)) {
      continue;
    }
    auto in_ctrl_nodes = out_ctrl_node->GetInControlNodes();
    if (in_ctrl_nodes.size() != kInCtrlNodesNum) {
      continue;
    }

    // Skip when has merge out
    bool has_merge_out = false;
    auto out_nodes_of_const = out_ctrl_node->GetOutAllNodes();
    for (const auto &out_node_of_const : out_nodes_of_const) {
      GE_CHECK_NOTNULL(out_node_of_const);
      if (out_node_of_const->GetType() == MERGE || out_node_of_const->GetType() == REFMERGE) {
        has_merge_out = true;
        break;
      }
    }
    if (has_merge_out) {
      continue;
    }

    GELOGI("Unlink control edge from %s to %s.", node->GetName().c_str(), out_ctrl_node->GetName().c_str());
    GE_CHK_GRAPH_STATUS_RET(out_ctrl_anchor->Unlink(out_ctrl_node->GetInControlAnchor()))
    for (auto &out_node_of_const : out_nodes_of_const) {
      if (!out_ctrl_anchor->IsLinkedWith(out_node_of_const->GetInControlAnchor())) {
        GELOGI("Link control edge from %s to %s.", node->GetName().c_str(), out_node_of_const->GetName().c_str());
        GE_CHK_GRAPH_STATUS_RET(out_ctrl_anchor->LinkTo(out_node_of_const->GetInControlAnchor()))
      }
    }
  }
  return SUCCESS;
}
}  // namespace ge
