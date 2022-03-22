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

#include "graph/passes/resource_pair_add_control_pass.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_adapter.h"

namespace {
const std::map<std::string, std::string> kResourcePairType = {{"StackPush", "StackPop"}};
const std::set<std::string> kResourceTypes = {"StackPush", "StackPop"};
}  // namespace

namespace ge {
Status ResourcePairAddControlPass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  GELOGD("ResourcePairAddControlPass pass start.");
  std::map<std::string, std::map<std::string, NodePtr>> prefix_2_node_per_type;
  // find all node of condition type, store with type and scope prefix key
  for (auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    auto node_type = node->GetType();
    if (kResourceTypes.find(node_type) != kResourceTypes.end()) {
      std::string node_name = node->GetName();
      std::string node_key(node_name);
      std::size_t found = node_name.rfind(node_type);
      if (found != std::string::npos) {
        node_key.replace(found, node_type.size(), "");
      }
      prefix_2_node_per_type[node_type][node_key] = node;
      GELOGD("ResourcePairAddControlPass insert node_key:%s, op_name:%s, op_type:%s", node_key.c_str(),
             node_name.c_str(), node->GetType().c_str());
    }
  }

  // according type pair, find same prefix node, add control edge
  for (auto &resource_type_pair : kResourcePairType) {
    auto from_item_prefix_2_node = prefix_2_node_per_type.find(resource_type_pair.first);
    if (from_item_prefix_2_node != prefix_2_node_per_type.end()) {
      for (auto &prefix_2_node : from_item_prefix_2_node->second) {
        const std::string &prefix = prefix_2_node.first;
        NodePtr from_node = prefix_2_node.second;
        GE_CHECK_NOTNULL(from_node);
        auto to_item_prefix_2_node = prefix_2_node_per_type.find(resource_type_pair.second);
        // stackpush and stackpop may exist in two subgraphs, no necessary to report error
        if (to_item_prefix_2_node == prefix_2_node_per_type.end()) {
          GELOGW("find peer type node fail, suffix:%s, from_type:%s, to_type:%s", prefix.c_str(),
                 resource_type_pair.first.c_str(), resource_type_pair.second.c_str());
          continue;
        }
        auto to_prefix_2_node = to_item_prefix_2_node->second.find(prefix);
        if (to_prefix_2_node == to_item_prefix_2_node->second.end()) {
          GELOGW("find peer prefix node fail, suffix:%s, from_type:%s, to_type:%s", prefix.c_str(),
                 resource_type_pair.first.c_str(), resource_type_pair.second.c_str());
          continue;
        }
        NodePtr to_node = to_prefix_2_node->second;
        GE_CHECK_NOTNULL(to_node);
        auto from_anchor = from_node->GetOutControlAnchor();
        auto to_anchor = to_node->GetInControlAnchor();
        GE_CHECK_NOTNULL(from_anchor);
        GE_CHECK_NOTNULL(to_anchor);
        graphStatus ret = from_anchor->LinkTo(to_anchor);
        if (ret != GRAPH_SUCCESS) {
          REPORT_CALL_ERROR("E19999", "Op:%s(%s) link control edge to op:%s(%s) failed",
                            from_node->GetName().c_str(), from_node->GetType().c_str(),
                            to_node->GetName().c_str(), to_node->GetType().c_str());
          GELOGE(PARAM_INVALID, "[Add][Edge] Op:%s(%s) link control edge to op:%s(%s) failed",
                 from_node->GetName().c_str(), from_node->GetType().c_str(),
                 to_node->GetName().c_str(), to_node->GetType().c_str());
          return PARAM_INVALID;
        }
        GELOGD("link success, from_node:%s, to_node:%s, from_type:%s, to_type:%s", from_node->GetName().c_str(),
               to_node->GetName().c_str(), resource_type_pair.first.c_str(), resource_type_pair.second.c_str());
      }
    }
  }
  return SUCCESS;
}
}  // namespace ge
