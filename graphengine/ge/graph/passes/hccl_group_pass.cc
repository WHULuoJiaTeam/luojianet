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

#include "graph/passes/hccl_group_pass.h"
#include <deque>
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/util.h"

namespace ge {
Status HcclGroupPass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  bool is_fused_node = false;
  if (!AttrUtils::GetBool(op_desc, ATTR_NAME_HCCL_FUSED_FLAG, is_fused_node)) {
    GELOGW("Get attr ATTR_NAME_GRADIENT_FUSED_GROUP failed.");
    return SUCCESS;
  }
  GELOGI("Recoginzed fused node %s", node->GetName().c_str());
  if (op_desc->HasAttr(ATTR_NAME_HCCL_FUSED_GROUP)) {
    GELOGD("Current node %s already marked group id, ignore it.", node->GetName().c_str());
    return SUCCESS;
  }

  if (!is_fused_node) {
    GELOGD("Current node %s is not gradient fused node , ignore it.", node->GetName().c_str());
    return SUCCESS;
  }
  Status ret = MarkGroupForFusedNode(node);
  if (ret != SUCCESS) {
    GELOGW("Mark group for fused node %s failed. It might cause performance problem.", node->GetName().c_str());
  }
  return SUCCESS;
}

Status HcclGroupPass::MarkGroupForFusedNode(NodePtr &fused_node) {
  std::deque<NodePtr> queue;
  queue.push_back(fused_node);
  string group_id = fused_node->GetName();

  while (!queue.empty()) {
    NodePtr node = queue.front();
    queue.pop_front();
    for (auto out_data_node : node->GetOutDataNodes()) {
      if (out_data_node->GetType() == fused_node->GetType()) {
        // if meet fused node, it is the end of current group
        break;
      }
      if (!AttrUtils::SetStr(out_data_node->GetOpDesc(), ATTR_NAME_HCCL_FUSED_GROUP, group_id)) {
        GELOGW("Set attr ATTR_NAME_GRADIENT_FUSED_GROUP failed.");
        return FAILED;
      }
      GELOGI("Set group_id %s for node %s", group_id.c_str(), out_data_node->GetName().c_str());
      queue.emplace_back(out_data_node);
    }
  }
  return SUCCESS;
}
}  // namespace ge
