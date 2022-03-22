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

#include "graph/passes/switch_logic_remove_pass.h"
#include <string>
#include <vector>
#include <utility>
#include "framework/common/debug/ge_log.h"
#include "graph/utils/graph_utils.h"
#include "graph/passes/pass_utils.h"
#include "framework/common/util.h"

namespace ge {
namespace {
using PredNodeAndOut = std::pair<NodePtr, int>;
constexpr int kSwitchOutputNum = 2;
constexpr int kSwitchPredIndex = 1;

char const *GetOutputNameFromIndex(int index) {
  if ((index >= 0) && (index < kSwitchOutputNum)) {
    static char const *name[kSwitchOutputNum] = {"false", "true"};
    return name[index];
  }
  return "UNKNOWN";
}

inline bool IsSwitch(const std::string &type) {
  return type == SWITCH || type == REFSWITCH;
}

Status GetPredNode(const NodePtr &switch_node, PredNodeAndOut &pred_node_index) {
  GE_CHECK_NOTNULL(switch_node);
  auto pred_in_anchor = switch_node->GetInDataAnchor(kSwitchPredIndex);
  if (pred_in_anchor == nullptr) {
    REPORT_INNER_ERROR("E19999", "Node:%s(%s) has no index:%d in data anchor, check invalid",
                       switch_node->GetName().c_str(), switch_node->GetType().c_str(), kSwitchPredIndex);
    GELOGE(INTERNAL_ERROR, "[Get][InDataAnchor] failed, Node:%s(%s) has no index:%d in data anchor",
           switch_node->GetName().c_str(), switch_node->GetType().c_str(), kSwitchPredIndex);
    return INTERNAL_ERROR;
  }
  auto pred_node_anchor = pred_in_anchor->GetPeerOutAnchor();
  if (pred_node_anchor == nullptr) {
    REPORT_INNER_ERROR("E19999", "Node:%s(%s)'s index:%d in data anchor, its peer anchor is nullptr, check invalid",
                       switch_node->GetName().c_str(), switch_node->GetType().c_str(), kSwitchPredIndex);
    GELOGE(INTERNAL_ERROR,
           "[Get][PeerOutAnchor] failed, Node:%s(%s)'s index:%d in data anchor, its peer anchor is nullptr",
           switch_node->GetName().c_str(), switch_node->GetType().c_str(), kSwitchPredIndex);
    return INTERNAL_ERROR;
  }
  auto pred_node = pred_node_anchor->GetOwnerNode();
  if (pred_node == nullptr) {
    REPORT_INNER_ERROR("E19999", "Node:%s(%s)'s index:%d in data anchor, its peer node is nullptr, check invalid",
                       switch_node->GetName().c_str(), switch_node->GetType().c_str(), kSwitchPredIndex);
    GELOGE(INTERNAL_ERROR,
           "[Get][OwnerNode] failed, Node:%s(%s)'s index:%d in data anchor, its peer node is nullptr",
           switch_node->GetName().c_str(), switch_node->GetType().c_str(), kSwitchPredIndex);
    return INTERNAL_ERROR;
  }
  pred_node_index.first = pred_node;
  pred_node_index.second = pred_node_anchor->GetIdx();
  return SUCCESS;
}
}  // namespace

Status SwitchLogicRemovePass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  if (!IsSwitch(node->GetType())) {
    return SUCCESS;
  }
  PredNodeAndOut pred_node_and_out;
  auto ret = GetPredNode(node, pred_node_and_out);
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Check][Param] Failed to run switch logic remove pass, no pred node found from switch %s",
           node->GetName().c_str());
    return INTERNAL_ERROR;
  }

  for (int i = 0; i < kSwitchOutputNum; ++i) {
    auto out_anchor = node->GetOutDataAnchor(i);
    if (out_anchor == nullptr) {
      GELOGW("Unexpected switch node, the %d out anchor is null", i);
      return SUCCESS;
    }
    for (auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
      if (in_anchor == nullptr) {
        REPORT_INNER_ERROR("E19999", "Node:%s(%s)'s index:%d out data anchor, its peer anchors has nullptr, "
                           "check invalid", node->GetName().c_str(), node->GetType().c_str(), i);
        GELOGE(INTERNAL_ERROR, "[Check][Param] Node:%s(%s)'s index:%d out data anchor, its peer anchors has nullptr",
               node->GetName().c_str(), node->GetType().c_str(), i);
        return INTERNAL_ERROR;
      }
      auto dst_node = in_anchor->GetOwnerNode();
      if (dst_node == nullptr) {
        REPORT_INNER_ERROR("E19999", "Node:%s(%s)'s index:%d out data anchor, its peer nodes has nullptr, "
                           "check invalid", node->GetName().c_str(), node->GetType().c_str(), i);
        GELOGE(INTERNAL_ERROR, "[Check][Param] Node:%s(%s)'s index:%d out data anchor, its peer nodes has nullptr",
               node->GetName().c_str(), node->GetType().c_str(), i);
        return INTERNAL_ERROR;
      }
      if (!IsSwitch(dst_node->GetType())) {
        continue;
      }
      PredNodeAndOut pred_node_next_switch;
      ret = GetPredNode(dst_node, pred_node_next_switch);
      if (ret != SUCCESS) {
        GELOGE(INTERNAL_ERROR,
               "[Check][Param] Failed to run switch logic remove pass, no pred node found from switch %s",
               dst_node->GetName().c_str());
        return INTERNAL_ERROR;
      }
      if (pred_node_and_out != pred_node_next_switch) {
        continue;
      }
      GELOGI("The switch nodes cascaded %s and %s have the save pred node %s, the %s can be remove",
             node->GetName().c_str(), dst_node->GetName().c_str(),
             pred_node_and_out.first->GetName().c_str(), dst_node->GetName().c_str());
      ret = RemoveSwitchNodeLogically(i, dst_node);
      if (ret != SUCCESS) {
        return ret;
      }
    }
  }

  return SUCCESS;
}

Status SwitchLogicRemovePass::RemoveSwitchNodeLogically(int parent_index, NodePtr &switch_node) {
  std::vector<int> isolate_map({-1, -1});
  for (int i = 0; i < kSwitchOutputNum; ++i) {
    if (i == parent_index) {
      isolate_map[i] = 0;
      continue;
    }
    GE_CHECK_NOTNULL(switch_node);
    auto out_anchor = switch_node->GetOutDataAnchor(i);
    if (out_anchor == nullptr) {
      GELOGW("The switch removing %s does not has %d out anchor, ignore it", switch_node->GetName().c_str(), i);
      continue;
    }

    GELOGI("Remove inactivate branch %s(%d) from switch %s",
           GetOutputNameFromIndex(i), i, switch_node->GetName().c_str());
    std::vector<NodePtr> deleted_nodes;
    std::vector<NodePtr> end_nodes;
    auto ret = PassUtils::RemoveInactiveBranchToMerge(out_anchor, deleted_nodes, end_nodes);
    if (ret != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Remove inactivate branch from node:%s(%s) to merge failed",
                        switch_node->GetName().c_str(), switch_node->GetType().c_str());
      GELOGE(FAILED, "[Remove][InactiveBranch] from node:%s(%s) to merge failed",
             switch_node->GetName().c_str(), switch_node->GetType().c_str());
      return ret;
    }

    for (auto &node : deleted_nodes) {
      GE_CHECK_NOTNULL(node);
      GELOGD("Remove node %s from inactivate branch from switch %s",
             node->GetName().c_str(), switch_node->GetName().c_str());
      AddNodeDeleted(node);
    }
    for (auto &node : end_nodes) {
      GE_CHECK_NOTNULL(node);
      GELOGD("Add end node %s to re-pass list, for inactivate branch from switch %s",
             node->GetName().c_str(), switch_node->GetName().c_str());
      AddRePassNode(node);
    }
  }
  GELOGI("Remove switch node cascaded %s, replace out index %d",
         switch_node->GetName().c_str(), parent_index);
  return IsolateAndDeleteNode(switch_node, isolate_map);
}
}  // namespace ge

