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

#include "compile_cache_state.h"
#include "framework/common/debug/ge_log.h"
namespace ge {
CacheItem CompileCacheState::GetNextCacheItem() {
  std::lock_guard<std::mutex> lock(cache_item_mu_);
  if (cache_item_queue_.empty()) {
    cache_item_queue_.push(cache_item_counter_++);
    return cache_item_counter_;
  } else {
    CacheItem next_item = cache_item_queue_.front();
    cache_item_queue_.pop();
    return next_item;
  }
}

void CompileCacheState::RecoveryCacheItem(const std::vector<CacheItem> &cache_items) {
  std::lock_guard<std::mutex> lock(cache_item_mu_);
  for (auto &item_id : cache_items) {
    cache_item_queue_.push(item_id);
  }
}

CacheItem CompileCacheState::AddCache(const CompileCacheDesc &compile_cache_desc) {
  CacheHashKey main_hash_key = CompileCacheHasher::GetCacheDescHashWithoutShape(compile_cache_desc);
  CacheHashKey shape_hash_key = CompileCacheHasher::GetCacheDescShapeHash(compile_cache_desc);
  std::lock_guard<std::mutex> lock(cc_state_mu_);
  auto iter = cc_state_.find(main_hash_key);
  if (iter == cc_state_.end()) {
    CacheItem next_item = GetNextCacheItem();
    CacheInfo cache_info = CacheInfo(std::time(nullptr), shape_hash_key, next_item, compile_cache_desc);
    std::vector<CacheInfo> info = {cache_info};
    cc_state_.insert({main_hash_key, info});
    return next_item;
  }
  auto &cached_item = iter->second;
  for (size_t idx = 0L; idx < cached_item.size(); idx++) {
    if ((cached_item[idx].shape_hash_ == shape_hash_key) &&
        (CompileCacheDesc::IsSameCompileDesc(cached_item[idx].desc_, compile_cache_desc))) {
      GELOGW("[AddCache] Same CompileCacheDesc has already been added, whose cache_item is %ld",
             cached_item[idx].item_);
      return cached_item[idx].item_;
    }
  }
  CacheItem next_item = GetNextCacheItem();
  CacheInfo cache_info = CacheInfo(std::time(nullptr), shape_hash_key, next_item, compile_cache_desc);
  cached_item.emplace_back(cache_info);
  return next_item;
}

std::vector<CacheItem> CompileCacheState::DelCache(const DelCacheFunc &func) {
  std::vector<CacheItem> delete_item;
  for (auto &item : cc_state_) {
    std::vector<CacheInfo> &cache_vec = item.second;
    for (auto iter = cache_vec.begin(); iter != cache_vec.end();) {
      if (func(*iter)) {
        delete_item.emplace_back((*iter).item_);
        std::lock_guard<std::mutex> lock(cc_state_mu_);
        iter = cache_vec.erase(iter);
      } else {
         iter++;
      }
    }
  }
  RecoveryCacheItem(delete_item);
  return delete_item;
}

std::vector<CacheItem> CompileCacheState::DelCache(const std::vector<CacheItem> &delete_item) {
  DelCacheFunc lamb = [&, delete_item] (CacheInfo &info) -> bool {
    auto iter = std::find(delete_item.begin(), delete_item.end(), info.GetItem());
    return iter != delete_item.end();
  };
  return DelCache(lamb);
}
}