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
#ifndef H39E4E719_91F4_4D0F_BA4F_6BA56CB1E20D
#define H39E4E719_91F4_4D0F_BA4F_6BA56CB1E20D

#include "fake_ns.h"
#include "common/opskernel/ops_kernel_builder.h"
#include "info_store_holder.h"

FAKE_NS_BEGIN

struct FakeOpsKernelBuilder : OpsKernelBuilder, InfoStoreHolder {
  FakeOpsKernelBuilder(const std::string &kernel_lib_name);
  FakeOpsKernelBuilder();

 private:
  Status Initialize(const map<std::string, std::string> &options) override;
  Status Finalize() override;
  Status CalcOpRunningParam(Node &node) override;
  Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) override;
};

FAKE_NS_END

#endif
