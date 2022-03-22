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

#include "ge_local_engine/ops_kernel_store/op/op_factory.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/op_desc.h"

namespace ge {
namespace ge_local {
OpFactory &OpFactory::Instance() {
  static OpFactory instance;
  return instance;
}

std::shared_ptr<Op> OpFactory::CreateOp(const Node &node, RunContext &run_context) {
  auto iter = op_creator_map_.find(node.GetType());
  if (iter != op_creator_map_.end()) {
    return iter->second(node, run_context);
  }
  REPORT_INNER_ERROR("E19999", "Not supported OP, type = %s, name = %s",
                     node.GetType().c_str(), node.GetName().c_str());
  GELOGE(FAILED, "[Check][Param] Not supported OP, type = %s, name = %s",
         node.GetType().c_str(), node.GetName().c_str());
  return nullptr;
}

void OpFactory::RegisterCreator(const std::string &type, const OP_CREATOR_FUNC &func) {
  if (func == nullptr) {
    GELOGW("Func is NULL.");
    return;
  }

  auto iter = op_creator_map_.find(type);
  if (iter != op_creator_map_.end()) {
    GELOGW("%s creator already exist", type.c_str());
    return;
  }

  op_creator_map_[type] = func;
  all_ops_.emplace_back(type);
}
}  // namespace ge_local
}  // namespace ge
