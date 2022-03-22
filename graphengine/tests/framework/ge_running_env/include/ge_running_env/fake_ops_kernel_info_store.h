/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef H1EBABA85_7056_48F0_B496_E4DB68E5FED3
#define H1EBABA85_7056_48F0_B496_E4DB68E5FED3

#include "fake_ns.h"
#include "common/opskernel/ops_kernel_info_store.h"
#include "ge/ge_api_types.h"
#include "info_store_holder.h"

FAKE_NS_BEGIN

struct FakeOpsKernelInfoStore : OpsKernelInfoStore, InfoStoreHolder {
  FakeOpsKernelInfoStore(const std::string &kernel_lib_name);
  FakeOpsKernelInfoStore();

 private:
  Status Initialize(const std::map<std::string, std::string> &options) override;
  Status Finalize() override;
  bool CheckSupported(const OpDescPtr &op_desc, std::string &reason) const override;
  void GetAllOpsKernelInfo(std::map<std::string, ge::OpInfo> &infos) const override;
};

FAKE_NS_END

#endif
