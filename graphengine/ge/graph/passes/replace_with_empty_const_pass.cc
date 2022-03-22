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

#include "graph/passes/replace_with_empty_const_pass.h"
#include <sstream>
#include <string>
#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"

namespace {
const std::unordered_set<std::string> kControlFlowOps = {
  ge::SWITCH,
  ge::REFSWITCH,
  ge::MERGE,
  ge::REFMERGE,
  ge::ENTER,
  ge::REFENTER,
  ge::NEXTITERATION,
  ge::REFNEXTITERATION,
  ge::EXIT,
  ge::REFEXIT,
  ge::LOOPCOND
};
}
namespace ge {
Status ReplaceWithEmptyConstPass::Run(NodePtr &node) {
  GELOGD("ReplaceWithEmptyConstPass in.");
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Check][Param] Parameter node is nullptr.");
    return PARAM_INVALID;
  }
  if (node->GetOpDesc() == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param node's op_desc is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Get][OpDesc] failed, Param [opDesc] must not be null.");
    return PARAM_INVALID;
  }
  if (node->GetType() == CONSTANT || node->GetType() == CONSTANTOP || node->GetType() == DATA) {
    GELOGI("Node %s is const. Ignore current pass.", node->GetName().c_str());
    return SUCCESS;
  }
  if (kControlFlowOps.count(NodeUtils::GetNodeType(node)) != 0) {
    GELOGI("Node %s is control flow op. Ignore current pass.", node->GetName().c_str());
    return SUCCESS;
  }
  // Node like no op, it has no output
  if (node->GetOpDesc()->GetAllOutputsDescPtr().empty()) {
    GELOGI("Node %s has no output desc. Ignore current pass.", node->GetName().c_str());
    return SUCCESS;
  }
  // If outputs of current node are all empty, replace it with empty const
  bool is_all_output_empty = true;
  for (const auto &output_desc_ptr : node->GetOpDesc()->GetAllOutputsDescPtr()) {
    if (output_desc_ptr == nullptr) {
      GELOGI("Node %s Got empty output_desc_ptr, ignore current pass.", node->GetName().c_str());
      return SUCCESS;
    }
    if (!IsKnownEmptyTenor(output_desc_ptr->GetShape())) {
      is_all_output_empty = false;
      break;
    }
  }
  if (is_all_output_empty) {
    GELOGI("Node %s has empty tensor output. It will be replaced by empty const.", node->GetName().c_str());
    // Replace op which all output is empty with empty const
    vector<GeTensorPtr> outputs;
    Status ret = GetOutputsOfCurrNode(node, outputs);
    if (ret != SUCCESS) {
      // If replace failed, it should not break whole process, so still return success
      GELOGW("Failed to get outputs of node %s.", node->GetName().c_str());
    }
    else {
      ret = Folding(node, outputs);
      if (ret != SUCCESS) {
        // If replace failed, it should not break whole process, so still return success
        GELOGW("Failed to repalce node %s with empty const.", node->GetName().c_str());
      }
    }
  }
  GELOGD("ReplaceWithEmptyConstPass end.");
  return SUCCESS;
}
Status ReplaceWithEmptyConstPass::GetOutputsOfCurrNode(const NodePtr &node_to_replace, vector<GeTensorPtr> &outputs) {
  for (const auto &out_anchor : node_to_replace->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(node_to_replace->GetOpDesc());
    auto out_desc = node_to_replace->GetOpDesc()->GetOutputDesc(out_anchor->GetIdx());
    GeTensorPtr empty_tensor = MakeShared<ge::GeTensor>(out_desc);
    GE_CHECK_NOTNULL(empty_tensor);
    outputs.emplace_back(empty_tensor);
  }
  return SUCCESS;
}

bool ReplaceWithEmptyConstPass::IsKnownEmptyTenor(const GeShape &shape) const {
  bool is_known_empty_tensor = false;
  for (auto dim : shape.GetDims()) {
    if (dim < 0) {
      // current dim is unknown dim, skip replace
      return false;
    } else if (dim == 0) {
      is_known_empty_tensor = true;
    }
  }
  return is_known_empty_tensor;
}
}  // namespace ge
