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

#include "ge_local_engine/ops_kernel_store/ge_local_ops_kernel_info.h"
#include <memory>
#include "ge_local_engine/common/constant/constant.h"
#include "common/ge/ge_util.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "ge_local_engine/ops_kernel_store/op/op_factory.h"
#include "proto/task.pb.h"

namespace ge {
namespace ge_local {
using domi::TaskDef;
using std::map;
using std::string;
using std::vector;

Status GeLocalOpsKernelInfoStore::Initialize(const map<string, string> &options) {
  GELOGI("GeLocalOpsKernelInfoStore init start.");

  OpInfo default_op_info = {.engine = kGeLocalEngineName,
                            .opKernelLib = kGeLocalOpKernelLibName,
                            .computeCost = 0,
                            .flagPartial = false,
                            .flagAsync = false,
                            .isAtomic = false};
  // Init op_info_map_
  auto all_ops = OpFactory::Instance().GetAllOps();
  for (auto &op : all_ops) {
    op_info_map_[op] = default_op_info;
  }

  GELOGI("GeLocalOpsKernelInfoStore inited success. op num=%zu", op_info_map_.size());

  return SUCCESS;
}

Status GeLocalOpsKernelInfoStore::Finalize() {
  op_info_map_.clear();
  return SUCCESS;
}

void GeLocalOpsKernelInfoStore::GetAllOpsKernelInfo(map<string, OpInfo> &infos) const { infos = op_info_map_; }

bool GeLocalOpsKernelInfoStore::CheckSupported(const OpDescPtr &op_desc, std::string &) const {
  if (op_desc == nullptr) {
    return false;
  }
  return op_info_map_.count(op_desc->GetType()) > 0;
}

Status GeLocalOpsKernelInfoStore::CreateSession(const map<string, string> &session_options) {
  // Do nothing
  return SUCCESS;
}

Status GeLocalOpsKernelInfoStore::DestroySession(const map<string, string> &session_options) {
  // Do nothing
  return SUCCESS;
}
}  // namespace ge_local
}  // namespace ge
