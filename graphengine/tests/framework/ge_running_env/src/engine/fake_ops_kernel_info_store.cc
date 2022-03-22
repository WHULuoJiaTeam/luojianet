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

#include "external/ge/ge_api_error_codes.h"
#include "ge_running_env/fake_ops_kernel_info_store.h"

FAKE_NS_BEGIN

FakeOpsKernelInfoStore::FakeOpsKernelInfoStore(const std::string &info_store_name) : InfoStoreHolder(info_store_name) {}

FakeOpsKernelInfoStore::FakeOpsKernelInfoStore() : InfoStoreHolder() {}

Status FakeOpsKernelInfoStore::Finalize() {
  op_info_map_.clear();
  return SUCCESS;
}

Status FakeOpsKernelInfoStore::Initialize(const std::map<std::string, std::string> &options) { return SUCCESS; }

void FakeOpsKernelInfoStore::GetAllOpsKernelInfo(map<string, OpInfo> &infos) const { infos = op_info_map_; }

bool FakeOpsKernelInfoStore::CheckSupported(const OpDescPtr &op_desc, std::string &) const {
  if (op_desc == nullptr) {
    return false;
  }
  return op_info_map_.count(op_desc->GetType()) > 0;
}

FAKE_NS_END
