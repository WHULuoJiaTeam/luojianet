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

#include "graph/passes/stop_gradient_pass.h"
#include <string>

namespace ge {
Status StopGradientPass::Run(NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param node is nullptr, check invalid");
    GELOGE(FAILED, "[Check][Param] parameter node is nullptr.");
    return FAILED;
  }
  string type;
  Status status_ret = GetOriginalType(node, type);
  if (status_ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Get OriginalType of op:%s(%s) failed",
                      node->GetName().c_str(), node->GetType().c_str());
    GELOGE(status_ret, "[Get][OriginalType] of op:%s(%s) failed",
           node->GetName().c_str(), node->GetType().c_str());
    return status_ret;
  }

  if (type == STOPGRADIENT) {
    return IsolateAndDeleteNode(node, {0});
  }
  return SUCCESS;
}
}  // namespace ge
