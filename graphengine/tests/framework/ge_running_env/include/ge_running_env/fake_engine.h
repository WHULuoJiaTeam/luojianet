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
#ifndef HAF5E9BF2_752F_4E03_B0A5_E1B912A5FA24
#define HAF5E9BF2_752F_4E03_B0A5_E1B912A5FA24

#include <string>
#include "fake_ns.h"
#include "ge_running_env/env_installer.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "opskernel_manager/ops_kernel_manager.h"
#include "register/ops_kernel_builder_registry.h"
#include "fake_ops_kernel_builder.h"
#include "fake_ops_kernel_info_store.h"

FAKE_NS_BEGIN

using FakeOpsKernelBuilderPtr = std::shared_ptr<FakeOpsKernelBuilder>;
using FakeOpsKernelInfoStorePtr = std::shared_ptr<FakeOpsKernelInfoStore>;

struct FakeEngine : EnvInstaller {
  FakeEngine(const std::string& engine_name);
  FakeEngine& KernelBuilder(FakeOpsKernelBuilderPtr);
  FakeEngine& KernelInfoStore(FakeOpsKernelInfoStorePtr);
  FakeEngine& KernelInfoStore(const std::string&);

 private:
  void InstallTo(std::map<string, OpsKernelInfoStorePtr>&) const override;
  void InstallTo(std::map<string, OpsKernelBuilderPtr>&) const override;

 private:
  template <typename BasePtr, typename SubClass>
  void InstallFor(std::map<string, BasePtr>& maps, const std::map<std::string, std::shared_ptr<SubClass>>&) const;

 private:
  std::string engine_name_;
  std::set<std::string> info_store_names_;
  std::map<std::string, FakeOpsKernelBuilderPtr> custom_builders_;
  std::map<std::string, FakeOpsKernelInfoStorePtr> custom_info_stores_;
};

FAKE_NS_END

#endif
