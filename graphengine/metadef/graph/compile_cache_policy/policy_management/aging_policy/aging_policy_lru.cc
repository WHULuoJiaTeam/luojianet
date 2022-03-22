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
#include "aging_policy_lru.h"
namespace ge {
std::vector<CacheItem> AgingPolicyLru::DoAging(const CCStatType &cc_state) const {
  std::vector<CacheItem> delete_item;
  int64_t delete_limit = std::time(nullptr) - delete_interval_;
  for (const auto &cache_item : cc_state) {
    const std::vector<CacheInfo> &cache_vec = cache_item.second;
    for (auto iter = cache_vec.begin(); iter != cache_vec.end(); iter++) {
      if ((*iter).GetTimeStamp() <= delete_limit) {
          delete_item.emplace_back((*iter).GetItem());
      }
    }
  }
  return delete_item;
}
}