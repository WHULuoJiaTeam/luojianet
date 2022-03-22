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

#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"
#include <map>
#include <string>
#include <vector>

namespace fe {
BufferFusionPassBase::BufferFusionPassBase() {}

BufferFusionPassBase::~BufferFusionPassBase() {}

Status BufferFusionPassBase::GetFusionNodes(const BufferFusionMapping &mapping,
                                            std::vector<ge::NodePtr> &fusion_nodes) {
  fusion_nodes = GetMatchedNodes(mapping);
  return SUCCESS;
}

Status BufferFusionPassBase::CalcFusionOpSliceInfo(vector<ge::NodePtr> &fusion_nodes, OpCalcInfo &op_slice_info) {
  return SUCCESS;
}

std::vector<ge::NodePtr> BufferFusionPassBase::GetMatchedNodes(const BufferFusionMapping &mapping) {
  std::vector<ge::NodePtr> nodes;
  for (const auto &item : mapping) {
    for (const auto &node : item.second) {
      nodes.push_back(node);
    }
  }
  return nodes;
}

std::vector<ge::NodePtr> BufferFusionPassBase::GetMatchedNodesByDescName(const std::string &desc_name,
                                                                         const BufferFusionMapping &mapping) {
  std::vector<ge::NodePtr> nodes;
  for (const auto &item : mapping) {
    const BufferFusionOpDesc *const op_desc = item.first;
    if ((op_desc != nullptr) && (op_desc->desc_name == desc_name)) {
      for (const auto &node : item.second) {
        nodes.push_back(node);
      }
    }
  }
  return nodes;
}

ge::NodePtr BufferFusionPassBase::GetMatchedHeadNode(const std::vector<ge::NodePtr> &matched_nodes) {
  for (const auto &node : matched_nodes) {
    const auto input_nodes = node->GetInDataNodes();
    bool find_flag = false;
    for (const auto &in_node : input_nodes) {
      // find the node from fuison sub graph
      if (std::find(matched_nodes.begin(), matched_nodes.end(), in_node) != matched_nodes.end()) {
        find_flag = true;
        break;
      }
    }
    if (find_flag == false) {
      return node;
    }
  }
  return nullptr;
}

}  // namespace fe
