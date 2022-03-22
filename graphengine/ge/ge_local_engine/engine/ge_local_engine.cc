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

#include "ge_local_engine/engine/ge_local_engine.h"
#include <map>
#include <memory>
#include <string>
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "ge_local_engine/common/constant/constant.h"
#include "common/ge/ge_util.h"
#include "ge_local_engine/ops_kernel_store/ge_local_ops_kernel_info.h"

namespace ge {
namespace ge_local {
GeLocalEngine &GeLocalEngine::Instance() {
  static GeLocalEngine instance;
  return instance;
}

Status GeLocalEngine::Initialize(const std::map<string, string> &options) {
  if (ops_kernel_store_ == nullptr) {
    ops_kernel_store_ = MakeShared<GeLocalOpsKernelInfoStore>();
    if (ops_kernel_store_ == nullptr) {
      REPORT_CALL_ERROR("E19999", "create GeLocalOpsKernelInfoStore failed.");
      GELOGE(FAILED, "[Call][MakeShared] Make GeLocalOpsKernelInfoStore failed.");
      return FAILED;
    }
  }
  return SUCCESS;
}

void GeLocalEngine::GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &ops_kernel_map) {
  if (ops_kernel_store_ != nullptr) {
    // add buildin opsKernel to opsKernelInfoMap
    ops_kernel_map[kGeLocalOpKernelLibName] = ops_kernel_store_;
  }
}

void GeLocalEngine::GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr> &) {
  // no optimizer for ge local engine
}

Status GeLocalEngine::Finalize() {
  ops_kernel_store_ = nullptr;
  return SUCCESS;
}
}  // namespace ge_local
}  // namespace ge

ge::Status Initialize(const std::map<string, string> &options) {
  return ge::ge_local::GeLocalEngine::Instance().Initialize(options);
}

void GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &ops_kernel_map) {
  ge::ge_local::GeLocalEngine::Instance().GetOpsKernelInfoStores(ops_kernel_map);
}

void GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr> &graph_optimizers) {
  ge::ge_local::GeLocalEngine::Instance().GetGraphOptimizerObjs(graph_optimizers);
}

ge::Status Finalize() { return ge::ge_local::GeLocalEngine::Instance().Finalize(); }
