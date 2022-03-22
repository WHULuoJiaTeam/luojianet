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

#include "register/graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include <algorithm>
#include <map>
#include <mutex>
#include <utility>
#include <vector>
#include "graph/debug/ge_log.h"

namespace fe {
class FusionPassRegistry::FusionPassRegistryImpl {
 public:
  void RegisterPass(const GraphFusionPassType &pass_type, const std::string &pass_name,
                    FusionPassRegistry::CreateFn create_fn) {
    const std::lock_guard<std::mutex> my_lock(mu_);

    auto iter = create_fns_.find(pass_type);
    if (iter != create_fns_.end()) {
      create_fns_[pass_type][pass_name] = create_fn;
      GELOGD("GraphFusionPass[type=%d,name=%s]: the pass type already exists.", pass_type, pass_name.c_str());
      return;
    }

    std::map<std::string, FusionPassRegistry::CreateFn> create_fn_map;
    create_fn_map[pass_name] = create_fn;
    create_fns_[pass_type] = create_fn_map;
    GELOGD("GraphFusionPass[type=%d,name=%s]: the pass type does not exist.", pass_type, pass_name.c_str());
  }

  std::map<std::string, FusionPassRegistry::CreateFn> GetCreateFn(const GraphFusionPassType &pass_type) {
    const std::lock_guard<std::mutex> my_lock(mu_);
    auto iter = create_fns_.find(pass_type);
    if (iter == create_fns_.end()) {
      return std::map<std::string, FusionPassRegistry::CreateFn>{};
    }
    return iter->second;
  }

 private:
  std::mutex mu_;
  std::map<GraphFusionPassType, map<std::string, FusionPassRegistry::CreateFn>> create_fns_;
};

FusionPassRegistry::FusionPassRegistry() {
  impl_ = std::unique_ptr<FusionPassRegistryImpl>(new (std::nothrow) FusionPassRegistryImpl);
}

FusionPassRegistry::~FusionPassRegistry() {}

FusionPassRegistry &FusionPassRegistry::GetInstance() {
  static FusionPassRegistry instance;
  return instance;
}

void FusionPassRegistry::RegisterPass(const GraphFusionPassType &pass_type, const std::string &pass_name,
                                      CreateFn create_fn) const {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "[Check][Param]param impl is nullptr, GraphFusionPass[type=%d,name=%s]: "
           "failed to register the graph fusion pass",
           pass_type, pass_name.c_str());
    return;
  }
  impl_->RegisterPass(pass_type, pass_name, create_fn);
}

std::map<std::string, FusionPassRegistry::CreateFn> FusionPassRegistry::GetCreateFnByType(
    const GraphFusionPassType &pass_type) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "[Check][Param]param impl is nullptr, GraphFusionPass[type=%d]: "
           "failed to create the graph fusion pass.", pass_type);
    return std::map<std::string, CreateFn>{};
  }
  return impl_->GetCreateFn(pass_type);
}

FusionPassRegistrar::FusionPassRegistrar(const GraphFusionPassType &pass_type, const std::string &pass_name,
                                         GraphPass *(*create_fn)()) {
  if (pass_type < BUILT_IN_GRAPH_PASS || pass_type >= GRAPH_FUSION_PASS_TYPE_RESERVED) {
    GELOGE(ge::PARAM_INVALID, "[Check][Param:pass_type] value:%d is not supported.", pass_type);
    return;
  }

  if (pass_name.empty()) {
    GELOGE(ge::PARAM_INVALID, "[Check][Param:pass_name]Failed to register the graph fusion pass, "
           "param pass_name is empty.");
    return;
  }
  FusionPassRegistry::GetInstance().RegisterPass(pass_type, pass_name, create_fn);
}

}  // namespace fe