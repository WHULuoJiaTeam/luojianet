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

#include "graph/manager/session_scope_mem_allocator.h"

#include <set>
#include <string>
#include <utility>

#include "framework/common/debug/ge_log.h"
#include "graph/manager/graph_mem_manager.h"

namespace ge {

SessionScopeMemAllocator::SessionScopeMemAllocator(rtMemType_t memory_type)
    : memory_type_(memory_type), memory_allocator_(nullptr) {}

Status SessionScopeMemAllocator::Initialize(uint32_t device_id) {
  GELOGI("Device id %u", device_id);
  // when redo Initialize free old memory
  FreeAllMemory();
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  memory_allocator_ = &MemManager::Instance().MemInstance(memory_type_);
  if (memory_allocator_ == nullptr) {
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  return ge::SUCCESS;
}

void SessionScopeMemAllocator::Finalize(uint32_t device_id) {
  GELOGI("Device id %u", device_id);
  FreeAllMemory();
}

uint8_t *SessionScopeMemAllocator::Malloc(size_t size, uint64_t session_id, uint32_t device_id) {
  GELOGI("Start malloc memory, size:%zu, session id:%lu device id:%u", size, session_id, device_id);
  const std::string purpose = "Memory for session scope.";
  auto ptr = memory_allocator_->MallocMemory(purpose, size, device_id);
  if (ptr == nullptr) {
    GELOGE(ge::FAILED, "Malloc failed, no enough memory for size:%zu, session_id:%lu device_id:%u", size,
           session_id, device_id);
    return nullptr;
  }
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  std::shared_ptr<uint8_t> mem_ptr(ptr, [&](uint8_t *p) { (void)memory_allocator_->FreeMemory(p); });
  allocated_memory_[session_id].emplace_back(size, mem_ptr);
  return ptr;
}

Status SessionScopeMemAllocator::Free(uint64_t session_id, uint32_t device_id) {
  GELOGI("Free session:%lu memory, device id:%u.", session_id, device_id);
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = allocated_memory_.find(session_id);
  if (it == allocated_memory_.end()) {
    GELOGW("Invalid session_id");
    return ge::PARAM_INVALID;
  }
  allocated_memory_.erase(it);
  return ge::SUCCESS;
}

void SessionScopeMemAllocator::FreeAllMemory() {
  GELOGI("Free all memory");
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  for (auto &session_mem : allocated_memory_) {
    session_mem.second.clear();
  }
  allocated_memory_.clear();
}
}  // namespace ge
