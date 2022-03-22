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

#include "register/ops_kernel_builder_registry.h"
#include "graph/debug/ge_log.h"

namespace ge {
OpsKernelBuilderRegistry::~OpsKernelBuilderRegistry() {
  for (auto &it : kernel_builders_) {
    GELOGW("[Unregister][Destruct] %s was not unregistered", it.first.c_str());
    // to avoid core dump when unregister is not called when so was close
    // this is called only when app is shutting down, so no release would be leaked
    new (std::nothrow) std::shared_ptr<OpsKernelBuilder>(it.second);
  }
}
void OpsKernelBuilderRegistry::Register(const string &lib_name, const OpsKernelBuilderPtr &instance) {
  const auto it = kernel_builders_.emplace(lib_name, instance);
  if (it.second) {
    GELOGI("Done registering OpsKernelBuilder successfully, kernel lib name = %s", lib_name.c_str());
  } else {
    GELOGW("[Register][Check] OpsKernelBuilder already registered. kernel lib name = %s", lib_name.c_str());
  }
}

void OpsKernelBuilderRegistry::UnregisterAll() {
  kernel_builders_.clear();
  GELOGI("All builders are unregistered");
}

void OpsKernelBuilderRegistry::Unregister(const string &lib_name) {
  (void)kernel_builders_.erase(lib_name);
  GELOGI("OpsKernelBuilder of %s is unregistered", lib_name.c_str());
}

const std::map<std::string, OpsKernelBuilderPtr> &OpsKernelBuilderRegistry::GetAll() const {
  return kernel_builders_;
}
OpsKernelBuilderRegistry &OpsKernelBuilderRegistry::GetInstance() {
  static OpsKernelBuilderRegistry instance;
  return instance;
}

OpsKernelBuilderRegistrar::OpsKernelBuilderRegistrar(const string &kernel_lib_name,
                                                     OpsKernelBuilderRegistrar::CreateFn const fn)
    : kernel_lib_name_(kernel_lib_name) {
  GELOGI("To register OpsKernelBuilder, kernel lib name = %s", kernel_lib_name.c_str());
  std::shared_ptr<OpsKernelBuilder> builder;
  if (fn != nullptr) {
    builder.reset(fn());
    if (builder == nullptr) {
      GELOGE(INTERNAL_ERROR, "[Create][OpsKernelBuilder]kernel lib name = %s", kernel_lib_name.c_str());
    }
  } else {
    GELOGE(INTERNAL_ERROR, "[Check][Param:fn]Creator is nullptr, kernel lib name = %s", kernel_lib_name.c_str());
  }

  // May add empty ptr, so that error can be found afterward
  OpsKernelBuilderRegistry::GetInstance().Register(kernel_lib_name, builder);
}

OpsKernelBuilderRegistrar::~OpsKernelBuilderRegistrar() {
  GELOGI("OpsKernelBuilderRegistrar destroyed. KernelLibName = %s", kernel_lib_name_.c_str());
  OpsKernelBuilderRegistry::GetInstance().Unregister(kernel_lib_name_);
}
}  // namespace ge
