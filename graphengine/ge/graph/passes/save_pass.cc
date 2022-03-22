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

#include "graph/passes/save_pass.h"

#include <string>
#include <utility>
#include <vector>
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/utils/graph_utils.h"

namespace ge {
namespace {
const char *const kSave = "Save";
const char *const kVar = "Variable";
const char *const kVarIsSave = "save_checkpoint";
const char *const kVarAttrVarIsSave = "_var_is_save";
}  // namespace

Status SavePass::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  vector<NodePtr> front_nodes;
  vector<uint8_t> out_index;
  vector<NodePtr> del_nodes;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kSave) {
      for (auto &in : node->GetAllInDataAnchors()) {
        auto out_anchor = in->GetPeerOutAnchor();
        if (out_anchor != nullptr) {
          ge::NodePtr peer_node = out_anchor->GetOwnerNode();
          if (peer_node->GetType() == kVar) {
            front_nodes.emplace_back(peer_node);
            out_index.emplace_back(out_anchor->GetIdx());
            ge::OpDescPtr op_desc = peer_node->GetOpDesc();
            GE_IF_BOOL_EXEC(!ge::AttrUtils::SetStr(op_desc, kVarAttrVarIsSave, kVarIsSave),
                            REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", kVarAttrVarIsSave,
                                              op_desc->GetName().c_str(), op_desc->GetType().c_str());
                            GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", kVarAttrVarIsSave,
                                   op_desc->GetName().c_str(), op_desc->GetType().c_str());
                            return INTERNAL_ERROR);
          }
        }
      }
      del_nodes.emplace_back(node);
    }
  }
  // add output nodes for save
  std::vector<std::pair<NodePtr, int32_t>> out_nodes_info{};
  for (size_t i = 0; i < front_nodes.size(); i++) {
    out_nodes_info.emplace_back(pair<NodePtr, int32_t>(front_nodes[i], out_index[i]));
  }
  graph->AppendGraphOutNodesInfo(out_nodes_info);

  // delete save node
  for (auto &node_ptr : del_nodes) {
    auto ret = graph->RemoveNode(node_ptr);
    if (ret != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Remove node:%s(%s) from graph:%s failed",
                        node_ptr->GetName().c_str(), node_ptr->GetType().c_str(), graph->GetName().c_str());
      GELOGE(ret, "[Remove][Node] %s(%s) from graph:%s failed",
             node_ptr->GetName().c_str(), node_ptr->GetType().c_str(), graph->GetName().c_str());
      return ret;
    }

    // update Target list
    vector<NodePtr> graph_target = graph->GetGraphTargetNodesInfo();
    auto iter = find(graph_target.begin(), graph_target.end(), node_ptr);
    if (iter != graph_target.end()) {
      GELOGI("Current node %s is as Target, remove it from target vector.", node_ptr->GetName().c_str());
      graph_target.erase(iter);
      graph->SetGraphTargetNodesInfo(graph_target);
    }
  }

  return SUCCESS;
}
}  // namespace ge
