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

#include "graph/passes/dropout_pass.h"

#include <string>

#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/utils/node_utils.h"

namespace ge {
///
/// run pass
/// @param [in] node node to be optimized
/// @return Status
///
Status DropOutPass::Run(NodePtr &node) {
  GELOGD("DropOutPass running");
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param node is nullptr, check invalid");
    GELOGE(FAILED, "[Check][Param] parameter node is nullptr.");
    return FAILED;
  }
  if (node->GetOpDesc() == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param op_desc of node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Get][OpDesc] failed, param [opDesc] must not be null.");
    return PARAM_INVALID;
  }
  std::string op_type = node->GetOpDesc()->GetType();
  if (op_type == DROPOUT) {
    GELOGD("op type is dropout.");
    return IsolateAndDeleteNode(node, {0});
  }
  return SUCCESS;
}
}  // namespace ge
