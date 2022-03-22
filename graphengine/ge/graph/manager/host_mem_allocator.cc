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

#include "graph/manager/host_mem_allocator.h"
#include "framework/common/debug/ge_log.h"
#include "common/ge/ge_util.h"

namespace ge {
const void *HostMemAllocator::Malloc(const std::shared_ptr<AlignedPtr> &aligned_ptr, size_t size) {
  if (aligned_ptr == nullptr) {
    GELOGW("Insert a null aligned_ptr, size=%zu", size);
    if (size == 0) {
      allocated_blocks_[nullptr] = { size, nullptr };
    }
    return nullptr;
  }
  GELOGD("allocate existed host memory succ, size=%zu", size);
  allocated_blocks_[aligned_ptr->Get()] = { size, aligned_ptr };
  return aligned_ptr->Get();
}

uint8_t *HostMemAllocator::Malloc(size_t size) {
  GELOGD("start to malloc host memory, size=%zu", size);
  std::lock_guard<std::mutex> lock(mutex_);
  std::shared_ptr<AlignedPtr> aligned_ptr = MakeShared<AlignedPtr>(size);
  if (aligned_ptr == nullptr) {
    REPORT_INNER_ERROR("E19999", "New AlignedPtr fail, size:%zu", size);
    GELOGE(INTERNAL_ERROR, "[Call][MakeShared] for AlignedPtr failed, size:%zu", size);
    return nullptr;
  }
  allocated_blocks_[aligned_ptr->Get()] = { size, aligned_ptr };
  GELOGD("allocate host memory succ, size=%zu", size);
  return aligned_ptr->MutableGet();
}

Status HostMemAllocator::Free(const void *memory_addr) {
  if (memory_addr == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param memory_addr is nullptr, check invalid");
    GELOGE(GE_GRAPH_FREE_FAILED, "[Check][Param] Invalid memory pointer");
    return GE_GRAPH_FREE_FAILED;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocated_blocks_.find(memory_addr);
  if (it == allocated_blocks_.end()) {
    REPORT_INNER_ERROR("E19999", "Memory_addr is not alloc before, check invalid");
    GELOGE(PARAM_INVALID, "[Check][Param] Invalid memory pointer:%p", memory_addr);
    return PARAM_INVALID;
  }
  it->second.second.reset();
  allocated_blocks_.erase(it);

  return SUCCESS;
}

void HostMemAllocator::Clear() {
  for (auto &block : allocated_blocks_) {
    block.second.second.reset();
  }
  allocated_blocks_.clear();
}
}  // namespace ge
