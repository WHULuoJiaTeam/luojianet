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

#include "graph/passes/print_op_pass.h"
#include <string>

namespace ge {
Status PrintOpPass::Run(ge::NodePtr &node) {
  GELOGD("PrintOpPass running");
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Check][Param] param [node] must not be null.");
    return PARAM_INVALID;
  }
  string type;
  Status ret = GetOriginalType(node, type);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][OriginalType] of node:%s failed", node->GetName().c_str());
    return ret;
  }
  if (type == "Print") {
    // print op has only one input and output data anchor
    return IsolateAndDeleteNode(node, {0});
  }
  return SUCCESS;
}
}  // namespace ge
