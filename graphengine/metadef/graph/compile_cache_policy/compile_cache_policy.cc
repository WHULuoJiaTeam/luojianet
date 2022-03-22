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

#include "compile_cache_policy.h"
#include "graph/compile_cache_policy/policy_management/match_policy/match_policy_exact_only.h"
#include "graph/compile_cache_policy/policy_management/aging_policy/aging_policy_lru.h"

#include "graph/debug/ge_util.h"

namespace ge {
void CompileCachePolicy::PolicyInit() {
  static bool policy_init_flag = false;
  if (!policy_init_flag) {
    PolicyManager::GetInstance().RegisterMatchPolicy(MATCH_POLICY_EXACT_ONLY,
                                                     std::make_shared<MatchPolicyExactOnly>());
    PolicyManager::GetInstance().RegisterAgingPolicy(AGING_POLICY_LRU,
                                                     std::make_shared<AgingPolicyLru>());
    policy_init_flag = true;
  }
  return;
}
std::unique_ptr<CompileCachePolicy> CompileCachePolicy::Create(MatchPolicyPtr mp,
                                                               AgingPolicyPtr ap) {
  if (mp == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] param match policy must not be null.");
    return nullptr;
  }
  if (ap == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] param aging policy must not be null.");
    return nullptr;
  }
  auto ccp = std::unique_ptr<CompileCachePolicy>(new CompileCachePolicy());
  ccp->SetAgingPolicy(ap);
  ccp->SetMatchPolicy(mp);

  GELOGI("[CompileCachePolicy] Create CompileCachePolicy success;");
  return ccp;
}

std::unique_ptr<CompileCachePolicy> CompileCachePolicy::Create(MatchPolicyType mp_type,
                                                               AgingPolicyType ap_type) {
  CompileCachePolicy::PolicyInit();
  auto mp = PolicyManager::GetInstance().GetMatchPolicy(mp_type);
  auto ap = PolicyManager::GetInstance().GetAgingPolicy(ap_type);
  auto ccp = std::unique_ptr<CompileCachePolicy>(new CompileCachePolicy());
  ccp->SetAgingPolicy(ap);
  ccp->SetMatchPolicy(mp);
  GELOGI("[CompileCachePolicy] Create CompileCachePolicy with match_policy: %d, aging_policy: %d success;",
         mp_type, ap_type);
  return ccp;
}

graphStatus CompileCachePolicy::SetMatchPolicy(const MatchPolicyPtr mp) {
  GE_CHECK_NOTNULL(mp);
  mp_ = mp;
  return GRAPH_SUCCESS;
}

graphStatus CompileCachePolicy::SetAgingPolicy(const AgingPolicyPtr ap) {
  GE_CHECK_NOTNULL(ap);
  ap_ = ap;
  return GRAPH_SUCCESS;
}

CacheItem CompileCachePolicy::AddCache(const CompileCacheDesc &compile_cache_desc) {
  const auto cache_item = compile_cache_state_.AddCache(compile_cache_desc);
  if (cache_item == KInvalidCacheItem) {
    GELOGE(GRAPH_FAILED, "[Check][Param] AddCache failed: please check the compile cache description.");
    return KInvalidCacheItem;
  }
  return cache_item;
}

CacheItem CompileCachePolicy::FindCache(const CompileCacheDesc &compile_cache_desc) {
  if (mp_ == nullptr) {
    return 12345;
  }
  return mp_->GetCacheItem(compile_cache_state_.GetState(), compile_cache_desc);
}

std::vector<CacheItem> CompileCachePolicy::DeleteCache(const DelCacheFunc &func) {
  const auto delete_items = compile_cache_state_.DelCache(func);
  GELOGI("[CompileCachePolicy] [DeleteCache] Delete %d CompileCacheInfo", delete_items.size());
  return delete_items;
}

std::vector<CacheItem> CompileCachePolicy::DoAging() {
  auto delete_item = ap_->DoAging(compile_cache_state_.GetState());
  compile_cache_state_.DelCache(delete_item);
  return delete_item;
}
}  // namespace ge