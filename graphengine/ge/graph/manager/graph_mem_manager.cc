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

#include "graph/manager/graph_mem_manager.h"

#include <string>

namespace ge {
MemManager::MemManager() {}

MemManager::~MemManager() { Finalize(); }

MemManager &MemManager::Instance() {
  static MemManager mem_manager;
  return mem_manager;
}

Status MemManager::Initialize(const std::vector<rtMemType_t> &memory_type) {
  std::lock_guard<std::recursive_mutex> lock(allocator_mutex_);
  if (init_) {
    GELOGW("MemManager has been inited.");
    return SUCCESS;
  }

  auto ret = InitAllocator(memory_type, memory_allocator_map_);
  if (ret != SUCCESS) {
    GELOGE(ret, "Create MemoryAllocator failed.");
    return ret;
  }

  ret = InitAllocator(memory_type, caching_allocator_map_);
  if (ret != SUCCESS) {
    GELOGE(ret, "Create CachingAllocator failed.");
    return ret;
  }

  ret = InitAllocator(memory_type, rdma_allocator_map_);
  if (ret != SUCCESS) {
    GELOGE(ret, "Create RdmaAllocator failed.");
    return ret;
  }

  ret = InitAllocator(memory_type, host_allocator_map_);
  if (ret != SUCCESS) {
    GELOGE(ret, "Create HostMemAllocator failed.");
    return ret;
  }

  ret = InitAllocator(memory_type, session_scope_allocator_map_);
  if (ret != SUCCESS) {
    GELOGE(ret, "Create HostMemAllocator failed.");
    return ret;
  }
  init_ = true;
  memory_type_ = memory_type;
  return SUCCESS;
}

template <typename T>
void FinalizeAllocatorMap(std::map<rtMemType_t, T *> &allocate_map) {
  for (auto &allocator : allocate_map) {
    if (allocator.second != nullptr) {
      allocator.second->Finalize();
      delete allocator.second;
      allocator.second = nullptr;
    }
  }
  allocate_map.clear();
}

void MemManager::Finalize() noexcept {
  GELOGI("Finalize.");
  std::lock_guard<std::recursive_mutex> lock(allocator_mutex_);
  // caching and rdma allocator use memory allocator, so finalize them first
  FinalizeAllocatorMap(session_scope_allocator_map_);
  FinalizeAllocatorMap(caching_allocator_map_);
  FinalizeAllocatorMap(rdma_allocator_map_);
  FinalizeAllocatorMap(host_allocator_map_);
  FinalizeAllocatorMap(memory_allocator_map_);
  init_ = false;
  memory_type_.clear();
}

MemoryAllocator &MemManager::MemInstance(rtMemType_t memory_type) {
  return GetAllocator(memory_type, memory_allocator_map_);
}

CachingAllocator &MemManager::CachingInstance(rtMemType_t memory_type) {
  return GetAllocator(memory_type, caching_allocator_map_);
}

RdmaPoolAllocator &MemManager::RdmaPoolInstance(rtMemType_t memory_type) {
  return GetAllocator(memory_type, rdma_allocator_map_);
}

HostMemAllocator &MemManager::HostMemInstance(rtMemType_t memory_type) {
  return GetAllocator(memory_type, host_allocator_map_);
}

SessionScopeMemAllocator &MemManager::SessionScopeMemInstance(rtMemType_t memory_type) {
  return GetAllocator(memory_type, session_scope_allocator_map_);
}
}  // namespace ge
