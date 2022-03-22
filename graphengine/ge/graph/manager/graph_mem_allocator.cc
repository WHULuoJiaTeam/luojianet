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

#include "graph/manager/graph_mem_allocator.h"

#include <string>

namespace ge {
Status MemoryAllocator::Initialize(uint32_t device_id) {
  GELOGI("MemoryAllocator::Initialize");

  // when redo Initialize free memory
  for (auto &it : memory_base_map_) {
    if (FreeMemory(it.second.memory_addr_, device_id) != ge::SUCCESS) {
      GELOGW("Initialize: FreeMemory failed");
    }
  }
  memory_base_map_.clear();
  return SUCCESS;
}

void MemoryAllocator::Finalize(uint32_t device_id) {
  GELOGI("MemoryAllocator::Finalize");

  // free memory
  for (auto &it : memory_base_map_) {
    if (FreeMemory(it.second.memory_addr_, device_id) != ge::SUCCESS) {
      GELOGW("Finalize: FreeMemory failed");
    }
  }
  memory_base_map_.clear();
}

uint8_t *MemoryAllocator::MallocMemory(const string &purpose, size_t memory_size, uint32_t device_id) const {
  uint8_t *memory_addr = nullptr;

  if (rtMalloc(reinterpret_cast<void **>(&memory_addr), memory_size, memory_type_) != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc fail, purpose:%s, size:%zu, device_id:%u",
                      purpose.c_str(), memory_size, device_id);
    GELOGE(ge::INTERNAL_ERROR, "[Malloc][Memory] failed, device_id = %u, size= %lu",
           device_id, memory_size);

    return nullptr;
  }

  GELOGI("MemoryAllocator::MallocMemory device_id = %u, size= %lu", device_id, memory_size);
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, purpose.c_str(), memory_size)
  return memory_addr;
}

Status MemoryAllocator::FreeMemory(uint8_t *memory_addr, uint32_t device_id) const {
  GELOGI("MemoryAllocator::FreeMemory device_id = %u", device_id);
  auto rtRet = rtFree(memory_addr);
  if (rtRet != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtFree fail, device_id:%u", device_id);
    GELOGE(rtRet, "[Call][RtFree] failed, device_id = %u", device_id);
    return RT_ERROR_TO_GE_STATUS(rtRet);
  }
  memory_addr = nullptr;
  return ge::SUCCESS;
}

uint8_t *MemoryAllocator::MallocMemory(const string &purpose, const string &memory_key, size_t memory_size,
                                       uint32_t device_id) {
  auto it = memory_base_map_.find(memory_key);
  if (it != memory_base_map_.end()) {
    it->second.memory_used_num_++;
    return it->second.memory_addr_;
  }

  uint8_t *memory_addr = MallocMemory(purpose, memory_size, device_id);

  if (memory_addr == nullptr) {
    REPORT_CALL_ERROR("E19999", "Malloc Memory fail, purpose:%s, memory_key:%s, memory_size:%zu, device_id:%u",
                      purpose.c_str(), memory_key.c_str(), memory_size, device_id);
    GELOGE(ge::INTERNAL_ERROR, "[Malloc][Memory] failed, memory_key[%s], size = %lu, device_id:%u.",
           memory_key.c_str(), memory_size, device_id);
    return nullptr;
  }

  MemoryInfo memory_info(memory_addr, memory_size);
  memory_info.memory_used_num_++;
  memory_base_map_[memory_key] = memory_info;
  mem_malloced_ = true;
  return memory_addr;
}

Status MemoryAllocator::FreeMemory(const string &memory_key, uint32_t device_id) {
  auto it = memory_base_map_.find(memory_key);
  if (it == memory_base_map_.end()) {
    if (mem_malloced_) {
      GELOGW(
          "MemoryAllocator::FreeMemory failed,"
          " memory_key[%s] was not exist, device_id = %u.",
          memory_key.c_str(), device_id);
    }
    return ge::INTERNAL_ERROR;
  }

  if (it->second.memory_used_num_ > 1) {
    GELOGW("MemoryAllocator::FreeMemory memory_key[%s] should not be released, reference count %d", memory_key.c_str(),
           it->second.memory_used_num_);
    // reference count greater than 1 represnt that static memory is used by
    // someone else, reference count decrement
    it->second.memory_used_num_--;
    return ge::SUCCESS;
  }

  if (FreeMemory(it->second.memory_addr_, device_id) != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Free Memory fail, memory_key:%s, device_id:%u",
                      memory_key.c_str(), device_id);
    GELOGE(ge::INTERNAL_ERROR, "[Free][Memory] failed, memory_key[%s], device_id:%u",
           memory_key.c_str(), device_id);
    return ge::INTERNAL_ERROR;
  }

  GELOGI("MemoryAllocator::FreeMemory device_id = %u", device_id);

  memory_base_map_.erase(it);
  return ge::SUCCESS;
}

uint8_t *MemoryAllocator::GetMemoryAddr(const string &memory_key, uint32_t device_id) {
  auto it = memory_base_map_.find(memory_key);
  if (it == memory_base_map_.end()) {
    GELOGW(
        "MemoryAllocator::GetMemoryAddr failed,"
        " memory_key[%s] was not exist, device_id = %u.",
        memory_key.c_str(), device_id);
    return nullptr;
  }

  return it->second.memory_addr_;
}
}  // namespace ge
