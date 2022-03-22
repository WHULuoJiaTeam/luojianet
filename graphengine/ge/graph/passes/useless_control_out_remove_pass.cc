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

#include "graph/passes/useless_control_out_remove_pass.h"

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"

namespace ge {
Status UselessControlOutRemovePass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);

  if ((node->GetType() != CONSTANT) && (node->GetType() != CONSTANTOP)) {
    return SUCCESS;
  }
  GELOGD("UselessControlOutRemovePass running, node: %s.", node->GetName().c_str());

  // const has no control input
  if (node->GetInControlNodes().empty()) {
    if (node->GetOutDataNodes().empty()) {
      // It is an isolated const, just remove it.
      GELOGI("Delete isolated const: %s.", node->GetName().c_str());
      GE_CHK_STATUS_RET(IsolateAndDeleteNode(node, {}))
      AddNodeDeleted(node);
    } else {
      auto out_ctrl_anchor = node->GetOutControlAnchor();
      if (out_ctrl_anchor != nullptr && !out_ctrl_anchor->GetPeerAnchors().empty()) {
        GELOGI("Node: %s unlink all out control edge.", node->GetName().c_str());
        out_ctrl_anchor->UnlinkAll();
      }
    }
  }

  return SUCCESS;
}
}  // namespace ge