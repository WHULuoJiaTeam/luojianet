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

#ifndef GE_GRAPH_MANAGER_SESSION_SCOPE_MEM_ALLOCATOR_H_
#define GE_GRAPH_MANAGER_SESSION_SCOPE_MEM_ALLOCATOR_H_

#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>

#include "framework/common/ge_inner_error_codes.h"
#include "graph/node.h"
#include "graph/manager/block_memory.h"
#include "runtime/mem.h"
#include "graph/manager/graph_mem_allocator.h"

namespace ge {
class SessionScopeMemoryInfo {
 public:
  SessionScopeMemoryInfo(size_t size, const std::shared_ptr<uint8_t> &ptr) : size(size), ptr(ptr) {}
  SessionScopeMemoryInfo() = delete;
  virtual ~SessionScopeMemoryInfo() = default;

  SessionScopeMemoryInfo(const SessionScopeMemoryInfo &other) {
    if (&other == this) {
      return;
    }
    size = other.size;
    ptr = other.ptr;
  };

  SessionScopeMemoryInfo &operator=(const SessionScopeMemoryInfo &other) {
    if (&other == this) {
      return *this;
    }
    size = other.size;
    ptr = other.ptr;
    return *this;
  };

 private:
  size_t size = 0;
  std::shared_ptr<uint8_t> ptr = nullptr;
};

class SessionScopeMemAllocator {
 public:
  explicit SessionScopeMemAllocator(rtMemType_t memory_type);

  SessionScopeMemAllocator(const SessionScopeMemAllocator &) = delete;

  SessionScopeMemAllocator &operator=(const SessionScopeMemAllocator &) = delete;

  virtual ~SessionScopeMemAllocator() = default;

  ///
  /// @ingroup ge_graph
  /// @brief caching allocator init
  /// @param [in] device id
  /// @return Status of init
  ///
  Status Initialize(uint32_t device_id = 0);

  ///
  /// @ingroup ge_graph
  /// @brief memory allocator finalize, release all memory
  /// @return void
  ///
  void Finalize(uint32_t device_id = 0);

  ///
  /// @ingroup ge_graph
  /// @brief malloc memory
  /// @param [in] size memory size
  /// @param [in] session_id session id
  /// @param [in] device id
  /// @return  memory address
  ///
  uint8_t *Malloc(size_t size, uint64_t session_id, uint32_t device_id = 0);

  ///
  /// @ingroup ge_graph
  /// @brief free memory
  /// @param [in] session_id session id
  /// @param [in] device_id device id
  /// @return Status result of function
  ///
  Status Free(uint64_t session_id, uint32_t device_id = 0);

 private:
  void FreeAllMemory();

 private:
  rtMemType_t memory_type_;

  // device memory allocator
  MemoryAllocator *memory_allocator_;

  // lock around all operations
  mutable std::recursive_mutex mutex_;

  // allocated blocks by memory pointer
  std::unordered_map<uint64_t, std::vector<SessionScopeMemoryInfo>> allocated_memory_;
};
}  // namespace ge
#endif  // GE_GRAPH_MANAGER_SESSION_SCOPE_MEM_ALLOCATOR_H_
