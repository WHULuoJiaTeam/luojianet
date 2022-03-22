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

#include "graph/passes/shape_operate_op_remove_pass.h"
#include "framework/common/debug/log.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/utils/attr_utils.h"

using domi::SUCCESS;

namespace ge {
Status ShapeOperateOpRemovePass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  for (auto &node : graph->GetDirectNode()) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(op_desc == nullptr, continue);
    bool to_be_deleted = false;
    GE_IF_BOOL_EXEC(!AttrUtils::GetBool(op_desc, ATTR_TO_BE_DELETED, to_be_deleted), to_be_deleted = false);
    GE_IF_BOOL_EXEC(to_be_deleted,
                    GE_CHK_STATUS_RET(graph->RemoveNode(node),
                                      "[Remove][Node] %s from graph:%s failed!", node->GetName().c_str(),
                                      graph->GetName().c_str()));
  }
  return SUCCESS;
}
}  // namespace ge
