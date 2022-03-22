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

#ifndef GRAPH_COMPILE_CACHE_POLICY_COMPILE_CACHE_STAT_H_
#define GRAPH_COMPILE_CACHE_POLICY_COMPILE_CACHE_STAT_H_

#include "compile_cache_desc.h"
#include "graph/ge_error_codes.h"
#include "compile_cache_hasher.h"
#include <vector>
#include <functional>
#include <unordered_map>
#include <ctime>
#include <queue>
#include <mutex>
namespace ge {
class CacheInfo;
using CacheItem = int64_t;
constexpr CacheItem KInvalidCacheItem = -1;

using DelCacheFunc = std::function<bool(CacheInfo &)>;
using CCStatType = std::unordered_map<uint64_t, std::vector<CacheInfo>>;

class CacheInfo {
friend class CompileCacheState;
public:
  CacheInfo(time_t time_stamp, CacheHashKey shape_hash, CacheItem item, const CompileCacheDesc &desc):
            time_stamp_(time_stamp), shape_hash_(shape_hash), item_(item), desc_(desc) {}
  CacheInfo(const CacheInfo &other) :
            time_stamp_(other.time_stamp_), shape_hash_(other.shape_hash_),
            item_(other.item_), desc_(other.desc_) {}
  CacheInfo &operator=(const CacheInfo &other) {
    time_stamp_ = other.time_stamp_;
    shape_hash_ = other.shape_hash_;
    item_ = other.item_;
    desc_ = other.desc_;
    return *this;
  }
  CacheInfo() = delete;
  ~CacheInfo() = default;

  void RefreshTimeStamp() {
    time_stamp_ = std::time(nullptr);
  }

  const time_t &GetTimeStamp() const noexcept {
    return time_stamp_;
  }

  const CacheHashKey &GetShapeHash() const noexcept {
    return shape_hash_;
  }

  const CacheItem &GetItem() const noexcept {
    return item_;
  }

  const CompileCacheDesc &GetCompileCacheDesc() const noexcept {
    return desc_;
  }

private:
  time_t time_stamp_;
  // hash combine result of shapes and origing shapes
  CacheHashKey shape_hash_;
  CacheItem item_;
  CompileCacheDesc desc_;
};

class CompileCacheState {
public:
  CompileCacheState() = default;
  ~CompileCacheState() = default;

  CacheItem AddCache(const CompileCacheDesc &compile_cache_desc);

  std::vector<CacheItem> DelCache(const DelCacheFunc &func);

  std::vector<CacheItem> DelCache(const std::vector<CacheItem> &delete_item);

  const CCStatType &GetState() const {
    return cc_state_;
  }

private:
  CacheItem GetNextCacheItem();
  void RecoveryCacheItem(const std::vector<CacheItem> &cache_items);

  std::mutex cc_state_mu_;
  std::mutex cache_item_mu_;

  int64_t cache_item_counter_ = 0L;
  std::queue<int64_t> cache_item_queue_;
  CCStatType cc_state_;
};
}  // namespace ge
#endif