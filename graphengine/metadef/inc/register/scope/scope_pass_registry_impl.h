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

#ifndef REGISTER_SCOPE_SCOPE_REGISTRY_IMPL_H_
#define REGISTER_SCOPE_SCOPE_REGISTRY_IMPL_H_

#include "external/register/scope/scope_fusion_pass_register.h"
#include <mutex>

namespace ge {
struct CreatePassFnPack;
class ScopeFusionPassRegistry::ScopeFusionPassRegistryImpl {
 public:
  void RegisterScopeFusionPass(const std::string &pass_name, ScopeFusionPassRegistry::CreateFn create_fn,
                               bool is_general);
  ScopeFusionPassRegistry::CreateFn GetCreateFn(const std::string &pass_name);
  std::unique_ptr<ScopeBasePass> CreateScopeFusionPass(const std::string &pass_name);
  std::vector<std::string> GetAllRegisteredPasses();
  bool SetPassEnableFlag(const std::string pass_name, const bool flag);

 private:
  std::mutex mu_;
  std::vector<std::string> pass_names_;  // In the order of user registration
  std::map<std::string, CreatePassFnPack> create_fn_packs_;
};
}  // namespace ge
#endif  // REGISTER_SCOPE_SCOPE_REGISTRY_IMPL_H_