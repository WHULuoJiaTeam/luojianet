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
#include "graph/passes/remove_same_const_pass.h"

#include <sstream>
#include <string>
#include <set>

#include "common/base64.h"
#include "ge_local_engine/engine/host_cpu_engine.h"
#include "graph/utils/node_utils.h"

namespace ge {
namespace {
std::string GetCseKey(const NodePtr &node) {
  std::stringstream ss;
  ss << node->GetType() << "control-inputs-";
  std::set<std::string> control_in_node_names;
  for (auto &src_node : node->GetInControlNodes()) {
    control_in_node_names.insert(src_node->GetName());
  }
  for (auto &name : control_in_node_names) {
    ss << name << "-";
  }

  ss << "attrs-" << AttrUtils::GetAllAttrsStr(node->GetOpDesc());

  return ss.str();
}

bool IsConstType(const NodePtr &node) { return (node->GetType() == CONSTANT || node->GetType() == CONSTANTOP); }
}  // namespace
Status RemoveSameConstPass::Run(ComputeGraphPtr graph) {
  GELOGD("Begin to run RemoveSameConstPass on the graph");
  GE_CHECK_NOTNULL(graph);
  std::map<std::string, NodePtr> keys_to_node;
  for (const auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (!IsConstType(node)) {
      continue;
    }
    bool is_unknown = false;
    auto ret = NodeUtils::GetNodeUnknownShapeStatus(*node, is_unknown);
    if (ret != GRAPH_SUCCESS) {
      GELOGW("Get node unknown status failed, node name:%s, type:%s.",
             node->GetName().c_str(), node->GetType().c_str());
      continue;
    }
    if (is_unknown) {
      GELOGI("Current node %s, type %s is unknown shape which should be skip.",
             node->GetName().c_str(), node->GetType().c_str());
      continue;
    }
    auto key = GetCseKey(node);
    GELOGD("The const node %s cse key %s", node->GetName().c_str(), ge::base64::EncodeToBase64(key).c_str());
    auto iter = keys_to_node.find(key);
    if (iter == keys_to_node.end()) {
      keys_to_node[key] = node;
      continue;
    }

    if (node->GetAllOutDataAnchorsSize() != iter->second->GetAllOutDataAnchorsSize()) {
      GELOGW("The const node %s and %s have the same CSE key, but different output anchor count, skip to fusion them",
             iter->second->GetName().c_str(), node->GetName().c_str());
      continue;
    }

    std::vector<int> output_map(node->GetAllOutDataAnchorsSize());
    for (size_t i = 0; i < node->GetAllOutDataAnchorsSize(); ++i) {
      output_map[i] = i;
    }

    ret = GraphUtils::ReplaceNodeAnchors(iter->second, node, {}, output_map);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Replace node:%s(%s)'s anchor by node:%s(%s) failed",
                        node->GetName().c_str(), node->GetType().c_str(),
                        iter->second->GetName().c_str(), iter->second->GetType().c_str());
      GELOGE(INTERNAL_ERROR, "[Replace][Anchors] of node:%s(%s) by node:%s(%s) failed",
             node->GetName().c_str(), node->GetType().c_str(),
             iter->second->GetName().c_str(), iter->second->GetType().c_str());
      return INTERNAL_ERROR;
    }

    NodeUtils::UnlinkAll(*node);

    ret = GraphUtils::RemoveNodeWithoutRelink(graph, node);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Remove node:%s(%s) without relink in graph:%s failed",
                        node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
      GELOGE(INTERNAL_ERROR, "[Remove][Node] %s(%s) without relink in graph:%s failed",
             node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
      return INTERNAL_ERROR;
    }

    GELOGI("Remove const node %s by RemoveSameConstPass, replace it with node %s", node->GetName().c_str(),
           iter->second->GetName().c_str());
  }
  return SUCCESS;
}
}  // namespace ge
