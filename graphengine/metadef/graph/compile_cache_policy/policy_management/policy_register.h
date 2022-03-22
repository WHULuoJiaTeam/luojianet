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

#ifndef GRAPH_COMPILE_CACHE_POLICY_POLICY_MANAGEMENT_POLICY_REGISTER_H_
#define GRAPH_COMPILE_CACHE_POLICY_POLICY_MANAGEMENT_POLICY_REGISTER_H_
#include <map>
#include <mutex>
#include "match_policy/match_policy.h"
#include "aging_policy/aging_policy.h"
namespace ge {
using MatchPolicyPtr = std::shared_ptr<MatchPolicy>;
using AgingPolicyPtr = std::shared_ptr<AgingPolicy>;
enum MatchPolicyType {
  MATCH_POLICY_EXACT_ONLY = 0
};
enum AgingPolicyType {
  AGING_POLICY_LRU = 0
};

class PolicyManager {
public:
  ~PolicyManager() = default;
  PolicyManager(const PolicyManager&) = delete;
  PolicyManager &operator=(const PolicyManager &other) = delete;
  static PolicyManager &GetInstance() {
    static PolicyManager instance;
    return instance;
  }
  void RegisterMatchPolicy(const MatchPolicyType match_policy_type, const MatchPolicyPtr ptr) {
    std::lock_guard<std::mutex> lock(mu_);
    match_policy_registry_.emplace(match_policy_type, ptr);
    return;
  }

  void RegisterAgingPolicy(const AgingPolicyType aging_policy_type, const AgingPolicyPtr ptr) {
    std::lock_guard<std::mutex> lock(mu_);
    aging_policy_registry_.emplace(aging_policy_type, ptr);
  }

  MatchPolicyPtr GetMatchPolicy(const MatchPolicyType match_policy_type) {
    auto iter = match_policy_registry_.find(match_policy_type);
    if (iter != match_policy_registry_.end()) {
      auto &mp_ptr = iter->second;
      if (mp_ptr != nullptr) {
        return mp_ptr;
      } else {
        GELOGE(ge::GRAPH_FAILED,
               "[GetMatchPolicy] failed. Match policy type : %d was incorrectly registered",
               match_policy_type);
        return nullptr;
      }
    }
    GELOGE(ge::GRAPH_FAILED,
           "[GetMatchPolicy] failed. Match policy type : %d  has not been registered",
           match_policy_type);
    return nullptr;
  }
  AgingPolicyPtr GetAgingPolicy(const AgingPolicyType aging_policy_type) {
    auto iter = aging_policy_registry_.find(aging_policy_type);
    if (iter != aging_policy_registry_.end()) {
      auto &ap_ptr = iter->second;
      if (ap_ptr != nullptr) {
        return ap_ptr;
      } else {
        GELOGE(ge::GRAPH_FAILED,
               "[GetAgingPolicy] failed. Match policy type : %d was incorrectly registered",
               aging_policy_type);
        return nullptr;
      }
    }
    GELOGE(ge::GRAPH_FAILED,
           "[GetAgingPolicy] failed. Match policy type : %d  has not been registered",
           aging_policy_type);
    return nullptr;
  }
private:
  PolicyManager() = default;
  std::mutex mu_;
  std::map<MatchPolicyType, MatchPolicyPtr> match_policy_registry_;
  std::map<AgingPolicyType, AgingPolicyPtr> aging_policy_registry_;
};
} // namespace ge
#endif