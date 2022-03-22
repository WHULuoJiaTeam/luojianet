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

#include "graph/passes/reshape_remove_pass.h"

#include  <map>
#include  <string>

#include "framework/common/util.h"
#include "framework/common/types.h"
#include "graph/passes/pass_utils.h"
#include "graph/utils/node_utils.h"

namespace ge {
namespace {
const int kReshapeDataIndex = 0;
enum OpHashValue {
  kReshapeType = 0,
  kReformatType = 1,
  kOpNoDelete = -1
};

std::map<std::string, OpHashValue> kToBeDeleteOp = {
  {RESHAPE, kReshapeType},
  {REFORMAT, kReformatType}
};
}

Status ReshapeRemovePass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  int key = kToBeDeleteOp.find(node->GetType()) == kToBeDeleteOp.end() ? kOpNoDelete : kToBeDeleteOp[node->GetType()];
  switch (key) {
    case kReshapeType: {
      bool is_shape_unknown = false;
      if (NodeUtils::GetNodeUnknownShapeStatus(*node, is_shape_unknown) == GRAPH_SUCCESS) {
        if (is_shape_unknown) {
          GELOGI("op:%s is unknown shape, can not be deleted.",
                 node->GetName().c_str());
          return SUCCESS;
        }
      }
      break;
    }
    case kReformatType:
      break;
    default:
      return SUCCESS;
  }

  GELOGI("Remove %s node %s", node->GetType().c_str(), node->GetName().c_str());
  return IsolateAndDeleteNode(node, {kReshapeDataIndex});
}
}  // namespace ge
