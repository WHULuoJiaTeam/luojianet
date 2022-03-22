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

#include "graph/manager/rdma_pool_allocator.h"

#include <framework/common/debug/log.h>
#include "framework/common/debug/ge_log.h"
#include "graph/ge_context.h"
#include "runtime/dev.h"
#include "graph/manager/graph_mem_manager.h"

namespace {
const size_t kAlignedSize = 512;
const float kSplitThreshold = 0.5;

inline size_t GetAlignedBlockSize(size_t size) {
  if (size == 0) {
    return kAlignedSize;
  }
  return kAlignedSize * ((size + kAlignedSize - 1) / kAlignedSize);
}

inline bool ShouldSplit(const ge::Block *block, size_t size) {
  return static_cast<double>(size) <= (static_cast<double>(block->size) * kSplitThreshold);
}

inline bool CanMerge(ge::Block *block) { return block != nullptr && !block->allocated; }
}  // namespace

namespace ge {
RdmaPoolAllocator::RdmaPoolAllocator(rtMemType_t memory_type)
    : memory_type_(memory_type), block_bin_(BlockBin([](const Block *left, const Block *right) {
        if (left->size != right->size) {
          return left->size < right->size;
        }
        return reinterpret_cast<uintptr_t>(left->ptr) < reinterpret_cast<uintptr_t>(right->ptr);
      })) {}

Status RdmaPoolAllocator::Initialize() {
  memory_allocator_ = &MemManager::Instance().MemInstance(memory_type_);
  if (memory_allocator_ == nullptr) {
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  return ge::SUCCESS;
}
void RdmaPoolAllocator::Finalize() {
  GELOGD("Rdma pool finalize start.");
  for (auto it = allocated_blocks_.begin(); it != allocated_blocks_.end();) {
    auto block = it->second;
    it = allocated_blocks_.erase(it);
    delete block;
  }
  for (auto it = block_bin_.begin(); it != block_bin_.end();) {
    auto block = *it;
    it = block_bin_.erase(it);
    delete block;
  }

  if (rdma_base_addr_ != nullptr) {
    GELOGD("Start to free rdma pool memory.");
    if (memory_allocator_->FreeMemory(rdma_base_addr_) != SUCCESS) {
      GELOGW("Free rdma pool memory failed");
    }
    rdma_base_addr_ = nullptr;
  }
}

Status RdmaPoolAllocator::InitMemory(size_t mem_size) {
  auto device_id = GetContext().DeviceId();
  GELOGD("Init Rdma Memory with size [%zu] for devid:[%u]", mem_size, device_id);
  if (rdma_base_addr_ != nullptr) {
    REPORT_INNER_ERROR("E19999", "Param rdma_base_addr_ is not nullptr, devid:%u, check invalid", device_id);
    GELOGE(GE_MULTI_INIT, "[Check][Param] Rdma pool has been malloced, devid:%u", device_id);
    return GE_MULTI_INIT;
  }
  const std::string purpose = "Memory for rdma pool.";
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto dev_id = static_cast<int32_t>(device_id);
  GE_CHK_RT_RET(rtSetDevice(dev_id));
  // DeviceReset before memory finished!
  GE_MAKE_GUARD(not_used_var, [&] { GE_CHK_RT(rtDeviceReset(dev_id)); });

  rdma_base_addr_ = memory_allocator_->MallocMemory(purpose, mem_size, device_id);
  if (rdma_base_addr_ == nullptr) {
    GELOGE(GE_GRAPH_MALLOC_FAILED, "[Malloc][Memory] failed, size:%zu, device_id:%u", mem_size, device_id);
    return GE_GRAPH_MALLOC_FAILED;
  }
  rdma_mem_size_ = mem_size;
  // Init with a base block.
  auto *base_block = new (std::nothrow) Block(device_id, mem_size, rdma_base_addr_);
  if (base_block == nullptr) {
    REPORT_CALL_ERROR("E19999", "New Block failed, size:%zu, device_id:%u", mem_size, device_id);
    GELOGE(GE_GRAPH_MALLOC_FAILED, "[New][Block] failed, size:%zu, device_id:%u", mem_size, device_id);
    return GE_GRAPH_MALLOC_FAILED;
  }
  block_bin_.insert(base_block);
  return SUCCESS;
}

uint8_t *RdmaPoolAllocator::Malloc(size_t size, uint32_t device_id) {
  GELOGI("start to malloc rdma memory size:%zu, device id = %u", size, device_id);
  auto aligned_size = GetAlignedBlockSize(size);
  Block key(device_id, aligned_size, nullptr);
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = block_bin_.lower_bound(&key);
  if (it != block_bin_.end()) {
    Block *block = *it;
    block_bin_.erase(it);
    block->allocated = true;
    if (block->ptr == nullptr) {
      REPORT_INNER_ERROR("E19999", "Rdmapool memory address is nullptr, device_id:%u, check invalid",
                         device_id);
      GELOGE(INTERNAL_ERROR, "[Check][Param] Rdmapool memory address is nullptr, device_id:%u", device_id);
      return nullptr;
    }
    allocated_blocks_.emplace(block->ptr, block);

    if (ShouldSplit(block, aligned_size)) {
      GELOGD("Block will be splited block size = %zu, aligned_size:%zu", block->size, aligned_size);
      auto *new_block =
          new (std::nothrow) Block(device_id, block->size - aligned_size, nullptr, block->ptr + aligned_size);
      if (new_block == nullptr) {
        GELOGW("Block split failed");
        return block->ptr;
      }
      new_block->next = block->next;
      if (block->next != nullptr) {
        block->next->prev = new_block;
      }
      new_block->prev = block;
      block->next = new_block;
      block->size = aligned_size;
      block_bin_.insert(new_block);
    }
    GELOGD("Find block size = %zu", block->size);
    return block->ptr;
  }
  GELOGW("Memory block not founded.");
  return nullptr;
}

Status RdmaPoolAllocator::Free(uint8_t *memory_addr, uint32_t device_id) {
  GELOGI("Free rdma memory, device id = %u", device_id);
  if (memory_addr == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param memory_addr is nullptr, device_id:%u, check invalid", device_id);
    GELOGE(GE_GRAPH_FREE_FAILED, "[Check][Param] Invalid memory pointer, device id:%u", device_id);
    return GE_GRAPH_FREE_FAILED;
  }

  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = allocated_blocks_.find(memory_addr);
  if (it == allocated_blocks_.end()) {
    REPORT_INNER_ERROR("E19999", "Param memory_addr is not allocated before, device_id:%u, "
                       "check invalid", device_id);
    GELOGE(PARAM_INVALID, "[Check][Param] Invalid memory pointer, device id:%u", device_id);
    return PARAM_INVALID;
  }

  Block *block = it->second;
  block->allocated = false;
  allocated_blocks_.erase(it);

  Block *merge_blocks[] = {block->prev, block->next};
  for (Block *merge_block : merge_blocks) {
    MergeBlocks(block, merge_block);
  }
  block_bin_.insert(block);

  return SUCCESS;
}

void RdmaPoolAllocator::MergeBlocks(Block *dst, Block *src) {
  if (!CanMerge(dst) || !CanMerge(src)) {
    return;
  }

  if (dst->prev == src) {
    dst->ptr = src->ptr;
    dst->prev = src->prev;
    if (dst->prev != nullptr) {
      dst->prev->next = dst;
    }
  } else {
    dst->next = src->next;
    if (dst->next != nullptr) {
      dst->next->prev = dst;
    }
  }

  dst->size += src->size;
  block_bin_.erase(src);
  delete src;
}

Status RdmaPoolAllocator::GetBaseAddr(uint64_t &base_addr, uint64_t &mem_size) {
  if (rdma_base_addr_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param rdma_base_addr_ is nullptr, check invalid");
    GELOGE(INTERNAL_ERROR, "[Check][Param] Rdma base addr is nullptr.");
    return INTERNAL_ERROR;
  }
  base_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rdma_base_addr_));
  mem_size = rdma_mem_size_;
  return SUCCESS;
}
}  // namespace ge
