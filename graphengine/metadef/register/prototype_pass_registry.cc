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

#include "register/prototype_pass_registry.h"

#include "graph/debug/ge_log.h"
#include "graph/types.h"

namespace ge {
class ProtoTypePassRegistry::ProtoTypePassRegistryImpl {
 public:
  void RegisterProtoTypePass(const std::string &pass_name, ProtoTypePassRegistry::CreateFn create_fn,
                             const domi::FrameworkType &fmk_type) {
    const std::lock_guard<std::mutex> lock(mu_);
    if (std::find(pass_names_.begin(), pass_names_.end(), pass_name) != pass_names_.end()) {
      GELOGW("[Register][Check] The prototype pass %s has been registered and will not overwrite the previous one",
             pass_name.c_str());
      return;
    }
    pass_names_.push_back(pass_name);

    const auto iter = create_fns_.find(fmk_type);
    if (iter != create_fns_.end()) {
      create_fns_[fmk_type].push_back(std::make_pair(pass_name, create_fn));
      GELOGD("Register prototype pass, pass name = %s", pass_name.c_str());
      return;
    }

    std::vector<std::pair<std::string, ProtoTypePassRegistry::CreateFn>> create_fn_vector;
    create_fn_vector.push_back(std::make_pair(pass_name, create_fn));
    create_fns_[fmk_type] = create_fn_vector;
    GELOGD("Register prototype pass, pass name = %s", pass_name.c_str());
  }

  std::vector<std::pair<std::string, ProtoTypePassRegistry::CreateFn>> GetCreateFnByType(
    domi::FrameworkType fmk_type) {
    const std::lock_guard<std::mutex> lock(mu_);
    const auto iter = create_fns_.find(fmk_type);
    if (iter == create_fns_.end()) {
      return std::vector<std::pair<std::string, ProtoTypePassRegistry::CreateFn>>{};
    }
    return iter->second;
  }

 private:
  std::mutex mu_;
  std::vector<std::string> pass_names_;
  std::map<domi::FrameworkType, std::vector<std::pair<std::string, ProtoTypePassRegistry::CreateFn>>> create_fns_;
};

ProtoTypePassRegistry::ProtoTypePassRegistry() {
  impl_ = std::unique_ptr<ProtoTypePassRegistryImpl>(new (std::nothrow) ProtoTypePassRegistryImpl);
}

ProtoTypePassRegistry::~ProtoTypePassRegistry() {}

ProtoTypePassRegistry &ProtoTypePassRegistry::GetInstance() {
  static ProtoTypePassRegistry instance;
  return instance;
}

void ProtoTypePassRegistry::RegisterProtoTypePass(const char_t *const pass_name, const CreateFn &create_fn,
                                                  const domi::FrameworkType &fmk_type) {
  if (impl_ == nullptr) {
    GELOGE(MEMALLOC_FAILED, "ProtoTypePassRegistry is not properly initialized.");
    return;
  }
  std::string str_pass_name;
  if (pass_name != nullptr) {
    str_pass_name = pass_name;
  }
  impl_->RegisterProtoTypePass(str_pass_name, create_fn, fmk_type);
}

std::vector<std::pair<std::string, ProtoTypePassRegistry::CreateFn>> ProtoTypePassRegistry::GetCreateFnByType(
    domi::FrameworkType fmk_type) const {
  if (impl_ == nullptr) {
    GELOGE(MEMALLOC_FAILED, "ProtoTypePassRegistry is not properly initialized.");
    return std::vector<std::pair<std::string, ProtoTypePassRegistry::CreateFn>>{};
  }
  return impl_->GetCreateFnByType(fmk_type);
}

ProtoTypePassRegistrar::ProtoTypePassRegistrar(const char_t *const pass_name, ProtoTypeBasePass *(*const create_fn)(),
                                               const domi::FrameworkType &fmk_type) {
  if (pass_name == nullptr) {
    GELOGE(PARAM_INVALID, "Failed to register ProtoType pass, pass name is null.");
    return;
  }
  ProtoTypePassRegistry::GetInstance().RegisterProtoTypePass(pass_name, create_fn, fmk_type);
}
}  // namespace ge
