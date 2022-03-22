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

#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include <algorithm>
#include <map>
#include <mutex>
#include <utility>
#include <vector>
#include "graph/debug/ge_log.h"

namespace fe {
class BufferFusionPassRegistry::BufferFusionPassRegistryImpl {
 public:
  void RegisterPass(const BufferFusionPassType &pass_type, const std::string &pass_name,
                    const BufferFusionPassRegistry::CreateFn create_fun) {
    const std::lock_guard<std::mutex> lock(mu_);
    const auto iter = create_fns_.find(pass_type);
    if (iter != create_fns_.end()) {
      create_fns_[pass_type][pass_name] = create_fun;
      GELOGI("UbFusionPass[type=%d,name=%s]: the pass type already exists.", pass_type, pass_name.c_str());
      return;
    }

    std::map<std::string, BufferFusionPassRegistry::CreateFn> create_fn_map;
    create_fn_map[pass_name] = create_fun;
    create_fns_[pass_type] = create_fn_map;
    GELOGI("UbFusionPass[type=%d,name=%s]: the pass type does not exists.", pass_type, pass_name.c_str());
  }

  std::map<std::string, BufferFusionPassRegistry::CreateFn> GetCreateFn(const BufferFusionPassType &pass_type) {
    std::lock_guard<std::mutex> lock(mu_);
    std::map<std::string, BufferFusionPassRegistry::CreateFn> result;
    auto iter = create_fns_.find(pass_type);
    if (iter == create_fns_.end()) {
      return result;
    }
    return iter->second;
  }

 private:
  std::mutex mu_;
  std::map<BufferFusionPassType, map<std::string, BufferFusionPassRegistry::CreateFn>> create_fns_;
};

BufferFusionPassRegistry::BufferFusionPassRegistry() {
  impl_ = std::unique_ptr<BufferFusionPassRegistryImpl>(new (std::nothrow) BufferFusionPassRegistryImpl);
}

BufferFusionPassRegistry::~BufferFusionPassRegistry() {}

BufferFusionPassRegistry &BufferFusionPassRegistry::GetInstance() {
  static BufferFusionPassRegistry instance;
  return instance;
}

void BufferFusionPassRegistry::RegisterPass(const BufferFusionPassType &pass_type, const std::string &pass_name,
                                            CreateFn create_fun) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "[Check][Param]UbFusionPass[type=%d,name=%s]: failed to register the ub fusion pass",
           pass_type, pass_name.c_str());
    return;
  }
  impl_->RegisterPass(pass_type, pass_name, create_fun);
}

std::map<std::string, BufferFusionPassRegistry::CreateFn> BufferFusionPassRegistry::GetCreateFnByType(
    const BufferFusionPassType &pass_type) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "[Check][Param]UbFusionPass[type=%d]: failed to create the ub fusion pass", pass_type);
    return std::map<std::string, CreateFn>{};
  }
  return impl_->GetCreateFn(pass_type);
}

BufferFusionPassRegistrar::BufferFusionPassRegistrar(const BufferFusionPassType &pass_type,
                                                     const std::string &pass_name,
                                                     BufferFusionPassBase *(*create_fun)()) {
  if ((pass_type < BUILT_IN_AI_CORE_BUFFER_FUSION_PASS) || (pass_type >= BUFFER_FUSION_PASS_TYPE_RESERVED)) {
    GELOGE(ge::PARAM_INVALID, "[Check][Param:pass_type] value %d is not supported.", pass_type);
    return;
  }

  if (pass_name.empty()) {
    GELOGE(ge::PARAM_INVALID, "[Check][Param:pass_name]Failed to register the ub fusion pass, the pass name is empty.");
    return;
  }

  BufferFusionPassRegistry::GetInstance().RegisterPass(pass_type, pass_name, create_fun);
}

}  // namespace fe