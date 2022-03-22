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

#include "ge_running_env/fake_engine.h"
#include "ge_running_env/fake_ops_kernel_builder.h"
#include "ge_running_env/fake_ops_kernel_info_store.h"
#include "opskernel_manager/ops_kernel_manager.h"

FAKE_NS_BEGIN

FakeEngine::FakeEngine(const std::string &engine_name) : engine_name_(engine_name) {}

FakeEngine &FakeEngine::KernelInfoStore(const std::string &info_store) {
  info_store_names_.insert(info_store);
  return *this;
}

FakeEngine &FakeEngine::KernelInfoStore(FakeOpsKernelInfoStorePtr ptr) {
  info_store_names_.insert(ptr->GetLibName());
  custom_info_stores_.insert(std::make_pair(ptr->GetLibName(), ptr));
  return *this;
}

FakeEngine &FakeEngine::KernelBuilder(FakeOpsKernelBuilderPtr builder) {
  info_store_names_.insert(builder->GetLibName());
  custom_builders_.insert(std::make_pair(builder->GetLibName(), builder));
  return *this;
}

namespace {
template <typename BasePtr, typename SubClass>
void InstallDefault(std::map<string, BasePtr> &maps, const std::string &info_store_name,
                    const std::string &engine_name) {
  auto parent_obj = std::make_shared<SubClass>(info_store_name);
  if (parent_obj == nullptr) {
    return;
  }
  parent_obj->EngineName(engine_name);
  maps.insert(std::make_pair(parent_obj->GetLibName(), parent_obj));
}
}  // namespace

template <typename BasePtr, typename SubClass>
void FakeEngine::InstallFor(std::map<string, BasePtr> &maps,
                            const std::map<std::string, std::shared_ptr<SubClass>> &child_maps) const {
  if (info_store_names_.empty()) {
    InstallDefault<BasePtr, SubClass>(maps, engine_name_, engine_name_);
  } else {
    for (auto &info_store_name : info_store_names_) {
      auto iter = child_maps.find(info_store_name);
      if (iter == child_maps.end()) {
        InstallDefault<BasePtr, SubClass>(maps, info_store_name, engine_name_);
      } else {
        maps.insert(std::make_pair(iter->second->GetLibName(), iter->second));
      }
    }
  }
}

void FakeEngine::InstallTo(std::map<string, OpsKernelInfoStorePtr> &ops_kernel_info_stores) const {
  InstallFor<OpsKernelInfoStorePtr, FakeOpsKernelInfoStore>(ops_kernel_info_stores, custom_info_stores_);
}

void FakeEngine::InstallTo(std::map<string, OpsKernelBuilderPtr> &ops_kernel_builders) const {
  InstallFor<OpsKernelBuilderPtr, FakeOpsKernelBuilder>(ops_kernel_builders, custom_builders_);
}

FAKE_NS_END
