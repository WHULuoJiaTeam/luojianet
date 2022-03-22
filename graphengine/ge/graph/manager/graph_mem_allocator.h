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

#ifndef GE_GRAPH_MANAGER_GRAPH_MEM_ALLOCATOR_H_
#define GE_GRAPH_MANAGER_GRAPH_MEM_ALLOCATOR_H_

#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/node.h"
#include "runtime/mem.h"

namespace ge {
class MemoryInfo {
 public:
  MemoryInfo() : memory_addr_(nullptr), memory_size_(0), memory_used_num_(0) {}

  MemoryInfo(uint8_t *memory_addr, size_t memory_size)
      : memory_addr_(memory_addr), memory_size_(memory_size), memory_used_num_(0) {}

  MemoryInfo &operator=(const MemoryInfo &op) {
    if (&op == this) {
      return *this;
    }

    this->memory_addr_ = op.memory_addr_;
    this->memory_size_ = op.memory_size_;
    this->memory_used_num_ = op.memory_used_num_;
    return *this;
  }

  MemoryInfo(const MemoryInfo &op) {
    this->memory_addr_ = op.memory_addr_;
    this->memory_size_ = op.memory_size_;
    this->memory_used_num_ = op.memory_used_num_;
  }
  virtual ~MemoryInfo() = default;

  uint8_t *memory_addr_;
  uint64_t memory_size_;
  int32_t memory_used_num_;
};

class MemoryAllocator {
 public:
  explicit MemoryAllocator(rtMemType_t memory_type) : memory_type_(memory_type), mem_malloced_(false) {}

  virtual ~MemoryAllocator() = default;

  ///
  /// @ingroup ge_graph
  /// @brief memory allocator init
  /// @param [in] options user config params
  /// @return Status of init
  ///
  Status Initialize(uint32_t device_id = 0);

  ///
  /// @ingroup ge_graph
  /// @brief memory allocator finalize
  /// @return void
  ///
  void Finalize(uint32_t device_id = 0);

  ///
  /// @ingroup ge_graph
  /// @brief malloc memory
  /// @param [in] purpose memory usage
  /// @param [in] size memory size
  /// @param [in] device_id device id
  /// @return  memory address
  ///
  uint8_t *MallocMemory(const string &purpose, size_t memory_size, uint32_t device_id = 0) const;

  ///
  /// @ingroup ge_graph
  /// @brief free memory
  /// @param [in] device_id device id
  /// @param [out] memory_ptr memory address ptr
  /// @return Status result of function
  ///
  Status FreeMemory(uint8_t *memory_addr, uint32_t device_id = 0) const;

  ///
  /// @ingroup ge_graph
  /// @brief malloc memory
  /// @param [in] purpose memory usage
  /// @param [in] memory_key memory key
  /// @param [in] size memory size
  /// @param [in] device_id device id
  /// @return memory address
  ///
  uint8_t *MallocMemory(const string &purpose, const string &memory_key, size_t memory_size,
                        uint32_t device_id = 0);

  ///
  /// @ingroup ge_graph
  /// @brief free memory
  /// @param [in] memory_key memory key
  /// @param [in] device_id device id
  /// @return Status result of function
  ///
  Status FreeMemory(const string &memory_key, uint32_t device_id = 0);

  ///
  /// @ingroup ge_graph
  /// @brief get memory address
  /// @param [in] memory_key memory key
  /// @param [in] device_id device id
  /// @return memory address (must not free memory by it)
  ///
  uint8_t *GetMemoryAddr(const string &memory_key, uint32_t device_id = 0);

 private:
  rtMemType_t memory_type_;
  bool mem_malloced_;
  map<string, MemoryInfo> memory_base_map_;
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_GRAPH_MEM_ALLOCATOR_H_
