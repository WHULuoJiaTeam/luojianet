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

#include "graph/passes/mark_node_unknown_shape_pass.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/local_context.h"

namespace ge {
namespace {
const char *const kEngineNameAiCore = "AIcoreEngine";
const char *const kNeedRefreshShape = "_need_generate";
const char *const kOriginalNode = "_original_node";
const int32_t kDynamicState = -2;
}

Status MarkNodeUnknownShapePass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  if (!GetLocalOmgContext().fuzz_compile_flag) {
    return SUCCESS;
  }
  if (IsAllAicoreSupportDyn(graph)) {
    if (UpdateNodeShapeToUnknown(graph) != SUCCESS) {
      GELOGE(FAILED, "[Update][Node_Shape]Failed to update node shape to unknown.");
      return FAILED;
    }
  }
  return SUCCESS;
}

bool MarkNodeUnknownShapePass::IsAllAicoreSupportDyn(ComputeGraphPtr &graph) {
  bool is_all_aicore_support_dyn = false;
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetOpDesc() == nullptr) {
      continue;
    }
    if (node->GetOpDesc()->GetOpKernelLibName() != kEngineNameAiCore) {
      GELOGD("Kernel of %s is %s.", node->GetName().c_str(), node->GetOpDesc()->GetOpKernelLibName().c_str());
      continue;
    }
    NodePtr original_node = nullptr;
    original_node = node->GetOpDesc()->TryGetExtAttr(kOriginalNode, original_node);
    if ((original_node == nullptr && AttrUtils::HasAttr(node->GetOpDesc(), ATTR_NAME_FUZZ_BUILD_RES_ATTRS)) ||
        (original_node != nullptr && AttrUtils::HasAttr(node->GetOpDesc(), ATTR_NAME_FUZZ_BUILD_RES_ATTRS) &&
        !AttrUtils::HasAttr(original_node->GetOpDesc(), kNeedRefreshShape))) {
      GELOGD("%s has set ATTR_NAME_FUZZ_BUILD_RES_ATTRS.", node->GetName().c_str());
      is_all_aicore_support_dyn = true;
    } else {
      GELOGD("%s has not set ATTR_NAME_FUZZ_BUILD_RES_ATTRS.", node->GetName().c_str());
      is_all_aicore_support_dyn = false;
      break;
    }
  }
  return is_all_aicore_support_dyn;
}

Status MarkNodeUnknownShapePass::UpdateNodeShapeToUnknown(ComputeGraphPtr &graph) {
  GELOGD("Need to update node shape to dynamic when get fuzz build result.");
  for (const auto &node : graph->GetAllNodes()) {
    if (NodeUtils::IsConst(*node) || node->GetType() == VARIABLE) {
      continue;
    }
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    for (size_t i = 0; i < op_desc->GetAllInputsSize(); ++i) {
      auto src_node = NodeUtils::GetInDataNodeByIndex(*node, static_cast<int>(i));
      if (src_node != nullptr && (NodeUtils::IsConst(*src_node) || src_node->GetType() == VARIABLE)) {
        continue;
      }
      GELOGD("Update input shape for %s.", node->GetName().c_str());
      auto input_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
      if (input_desc != nullptr) {
        input_desc->SetShape(GeShape({kDynamicState}));
      }
    }

    for (auto &output_desc : op_desc->GetAllOutputsDescPtr()) {
      if (output_desc != nullptr) {
        GELOGD("Update output shape for %s.", node->GetName().c_str());
        output_desc->SetShape(GeShape({kDynamicState}));
      }
    }
  }
  return SUCCESS;
}
}  // namespace ge