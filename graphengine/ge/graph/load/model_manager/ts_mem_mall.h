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

#ifndef GE_GRAPH_LOAD_TS_MEM_MALL_H_
#define GE_GRAPH_LOAD_TS_MEM_MALL_H_

#include <mutex>
#include <unordered_map>
#include <memory>

#include "runtime/base.h"
#include "framework/common/debug/ge_log.h"

namespace {
constexpr uint32_t kMaxTsMemBlock = 2097152;   // Max block 2M 2 * 1024 * 1024
constexpr uint32_t kTsMemAligment = 64;   // Malloc for 64 bits align
constexpr uint32_t kTsMemAlignMask = kTsMemAligment - 1;
}

namespace ge {
class TsMemMall {
 public:
  TsMemMall() {
    mem_type_ = RT_MEMORY_TS_4G;
  }
  TsMemMall(rtMemType_t type) {
    mem_type_ = type;
  }
  ~TsMemMall() {
    for (auto it : mem_store_size_) {
      rtError_t ret = rtFree(it.second);
      if (ret != RT_ERROR_NONE) {
        GELOGE(RT_FAILED, "[Call][RtFree] failed, ret:0x%X", ret);
      }
    }
    mem_store_size_.clear();
    mem_store_addr_.clear();
  }

  void *Acquire(int64_t offset, uint64_t size) {
    if (size == 0) {
      GELOGE(RT_FAILED, "[Check][Param] Acquire mem block failed, size:%lu", size);
      return nullptr;
    }

    uint64_t bytes = (size + kTsMemAlignMask) & ~kTsMemAlignMask;
    if (bytes > kMaxTsMemBlock) {
      GELOGW("Acquire TS memory may not physical continuity, size: %lu", bytes);
    }

    std::lock_guard<std::mutex> lock(mem_mutex_);
    const auto it = mem_store_size_.find(offset);
    if (it != mem_store_size_.end()) {
      GELOGI("Acquire TS memory: %p, offset: %ld, size: %lu, align: %lu", it->second, offset, size, bytes);
      return it->second;
    }

    void *addr = nullptr;
    rtError_t rt_ret = rtMalloc(&addr, bytes, mem_type_);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "[Call][RtMalloc] failed, size:%lu, ret:0x%X", bytes, rt_ret);
      return nullptr;
    }

    GELOGI("Acquire TS memory: %p, offset: %ld, size: %lu, align: %lu", addr, offset, size, bytes);
    mem_store_size_[offset] = addr;
    mem_store_addr_[addr] = offset;
    return addr;
  }

  void Release(void *addr) {
    std::lock_guard<std::mutex> lock(mem_mutex_);
    const auto it = mem_store_addr_.find(addr);
    if (it == mem_store_addr_.end()) {
      GELOGW("Not TS memory: %p.", addr);
      return;
    }

    GELOGI("Release TS memory: %p.", addr);
    mem_store_size_.erase(it->second);
    mem_store_addr_.erase(it);
    rtError_t ret = rtFree(addr);
    if (ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "[Call][RtFree] failed, ret:0x%X", ret);
    }
  }

 private:
  std::mutex mem_mutex_;
  std::map<int64_t, void *> mem_store_size_;
  std::map<void *, int64_t> mem_store_addr_;
  rtMemType_t mem_type_;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_TS_MEM_MALL_H_
