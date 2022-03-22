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
#include "graph/passes/reshape_recovery_pass.h"
#include "common/ge/ge_util.h"

namespace ge {
namespace {
NodePtr CreateReshape(const ConstGeTensorDescPtr &src, const ConstGeTensorDescPtr &dst, const ComputeGraphPtr &graph) {
  static std::atomic_long reshape_num(0);
  auto next_num = reshape_num.fetch_add(1);
  auto reshape = MakeShared<OpDesc>("Reshape_ReshapeRecoveryPass_" + std::to_string(next_num), RESHAPE);
  if (reshape == nullptr) {
    REPORT_CALL_ERROR("E19999", "New OpDesc failed");
    GELOGE(FAILED, "[New][OpDesc] failed");
    return nullptr;
  }
  auto ret = reshape->AddInputDesc("x", *src);
  if (ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed, name:x",
                      reshape->GetName().c_str(), reshape->GetType().c_str());
    GELOGE(FAILED, "[Add][InputDesc] to op:%s(%s) failed, name:x",
           reshape->GetName().c_str(), reshape->GetType().c_str());
    return nullptr;
  }
  ret = reshape->AddInputDesc("shape", GeTensorDesc(GeShape(), Format(), DT_INT32));
  if (ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed, name:shape",
                      reshape->GetName().c_str(), reshape->GetType().c_str());
    GELOGE(FAILED, "[Add][InputDesc] to op:%s(%s) failed, name:shape",
           reshape->GetName().c_str(), reshape->GetType().c_str());
    return nullptr;
  }
  ret = reshape->AddOutputDesc("y", *dst);
  if (ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add output desc to op:%s(%s) failed, name:y",
                      reshape->GetName().c_str(), reshape->GetType().c_str());
    GELOGE(FAILED, "[Add][OutputDesc] to op:%s(%s) failed, name:y",
           reshape->GetName().c_str(), reshape->GetType().c_str());
    return nullptr;
  }

  return graph->AddNode(reshape);
}

Status InsertReshapeIfNeed(const NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  for (auto src_anchor : node->GetAllOutDataAnchors()) {
    auto src_tensor = node->GetOpDesc()->GetOutputDescPtr(src_anchor->GetIdx());
    GE_CHECK_NOTNULL(src_tensor);
    for (auto dst_anchor : src_anchor->GetPeerInDataAnchors()) {
      auto dst_node = dst_anchor->GetOwnerNode();
      GELOGD("Try insert reshape between %s[%d] and %s[%d] to keep the shape continues",
             node->GetName().c_str(), src_anchor->GetIdx(), dst_node->GetName().c_str(), dst_anchor->GetIdx());
      GE_CHECK_NOTNULL(dst_node);
      GE_CHECK_NOTNULL(dst_node->GetOpDesc());
      auto dst_tensor = dst_node->GetOpDesc()->MutableInputDesc(dst_anchor->GetIdx());
      GE_CHECK_NOTNULL(dst_tensor);
      bool is_dynamic = false;
      const auto &src_tensor_dims = src_tensor->GetShape().GetDims();
      const auto &dst_tensor_dims = dst_tensor->GetShape().GetDims();
      if ((std::any_of(src_tensor_dims.begin(), src_tensor_dims.end(), [](int64_t val) { return val < 0 ; }))
          || (std::any_of(dst_tensor_dims.begin(), dst_tensor_dims.end(), [](int64_t val) { return val < 0; }))) {
        GELOGD("No need to insert reshape node between %s nad %s.", node->GetName().c_str(),
               dst_node->GetName().c_str());
        is_dynamic = true;
      }
      if (dst_node->GetType() == NETOUTPUT && is_dynamic) {
        // NetOutput shape must be continuous when dynamic shape.
        // Otherwise, there may be an error waiting for the shape refresh to time out during execution.
        dst_tensor->SetShape(src_tensor->GetShape());
        continue;
      }
      bool is_need_insert_reshape = src_tensor_dims != dst_tensor_dims &&
                                    !is_dynamic;
      if (is_need_insert_reshape) {
        auto reshape = CreateReshape(src_tensor, dst_tensor, node->GetOwnerComputeGraph());
        GE_CHECK_NOTNULL(reshape);
        auto ret = GraphUtils::InsertNodeBetweenDataAnchors(src_anchor, dst_anchor, reshape);
        if (ret != GRAPH_SUCCESS) {
          REPORT_CALL_ERROR("E19999",
                            "Insert node:%s(%s) between node:%s(%s)(out_index:%d) and node:%s(%s)(out_index:%d) failed",
                            reshape->GetName().c_str(), reshape->GetType().c_str(),
                            node->GetName().c_str(), node->GetType().c_str(), src_anchor->GetIdx(),
                            dst_node->GetName().c_str(), dst_node->GetType().c_str(), dst_anchor->GetIdx());
          GELOGE(INTERNAL_ERROR,
                 "[Insert][Node] %s(%s) between node:%s(%s)(out_index:%d) and node:%s(%s)(out_index:%d) failed",
                 reshape->GetName().c_str(), reshape->GetType().c_str(),
                 node->GetName().c_str(), node->GetType().c_str(), src_anchor->GetIdx(),
                 dst_node->GetName().c_str(), dst_node->GetType().c_str(), dst_anchor->GetIdx());
          return INTERNAL_ERROR;
        }
        GELOGI("Insert reshape between %s and %s to keep the shape continues",
               node->GetName().c_str(), dst_node->GetName().c_str());
      }
    }
  }
  return SUCCESS;
}
}  // namespace

Status ReshapeRecoveryPass::Run(ComputeGraphPtr graph) {
  for (const auto &node : graph->GetDirectNode()) {
    auto ret = InsertReshapeIfNeed(node);
    if (ret != SUCCESS) {
      return ret;
    }
  }
  return SUCCESS;
}
}  // namespace ge
