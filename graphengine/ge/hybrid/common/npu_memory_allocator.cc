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

#include "hybrid/common/npu_memory_allocator.h"
#include <mutex>
#include "framework/common/debug/log.h"
#include "graph/manager/graph_mem_manager.h"

namespace ge {
namespace hybrid {
const size_t kPaddingUnit = 2;

size_t kMaxHbmMemorySize = 1024UL * 1024UL * 1024UL * 1024UL; // 1024G

std::map<uint32_t, std::unique_ptr<NpuMemoryAllocator>> NpuMemoryAllocator::allocators_;
std::mutex NpuMemoryAllocator::mu_;

AllocationAttr::AllocationAttr(int padding, void *try_reuse_addr, MemStorageType mem_type)
    : padding_(padding), try_reuse_addr_(try_reuse_addr), mem_type_(mem_type) {}
AllocationAttr::AllocationAttr(int padding) : AllocationAttr(padding, nullptr) {}
AllocationAttr::AllocationAttr(void *try_reuse_addr) : AllocationAttr(0, try_reuse_addr) {}

NpuMemoryAllocator *NpuMemoryAllocator::GetAllocator() {
  int32_t device_id = 0;
  auto rt_result = rtGetDevice(&device_id);
  if (rt_result != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "[Get][Device] Failed, result:%d.", rt_result);
    REPORT_INNER_ERROR("E19999", "rtGetDevice failed, result:%d.", rt_result);
    return nullptr;
  }

  GELOGD("Got device id = %d from context", device_id);
  return GetAllocator(static_cast<uint32_t>(device_id));
}

NpuMemoryAllocator::NpuMemoryAllocator(uint32_t device_id) : device_id_(device_id) {}

void *NpuMemoryAllocator::Allocate(std::size_t size, AllocationAttr *attr) {
  size_t allocate_size = size;
  MemStorageType mem_type = HBM;
  if (attr != nullptr) {
    mem_type = attr->mem_type_;
  }

  if (allocate_size == 0) {
    GELOGE(MEMALLOC_FAILED, "[Check][Param:size_t]Memory size is 0, device_id = %u, size = %zu.",
        device_id_, allocate_size);
    REPORT_INNER_ERROR("E19999", "Memory size is 0, device_id = %u, size = %zu.", device_id_, allocate_size);
    return nullptr;
  }

  void *buffer = nullptr;
  if (mem_type == RDMA_HBM) {
    buffer = MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).Malloc(allocate_size, device_id_);
  } else if (mem_type == HOST_DDR) {
    buffer = MemManager::Instance().HostMemInstance(RT_MEMORY_HBM).Malloc(allocate_size);
  } else {
    if (allocate_size > kMaxHbmMemorySize) {
      GELOGE(PARAM_INVALID, "[Check][Param:size_t]Invalid HBM memory size: %zu bigger than limit:%lu, check invalid.",
             allocate_size, kMaxHbmMemorySize);
      REPORT_CALL_ERROR("E19999", "Invalid HBM memory size: %zu bigger than limit:%lu, check invalid.",
                        allocate_size, kMaxHbmMemorySize);
      return nullptr;
    }
    void *try_reuse_addr = nullptr;
    int padding = kDefaultPadding;
    if (attr != nullptr) {
      try_reuse_addr = attr->try_reuse_addr_;
      if (attr->padding_ > 0) {
        padding = attr->padding_;
      }
    }
    // padding up to multiple of padding, and add extra padding
    allocate_size = (size + kPaddingUnit * padding - 1) / padding * padding;
    GELOGD("Padding size %ld by %d. final size = %zu.", size, padding, allocate_size);
    buffer = MemManager::Instance()
                 .CachingInstance(RT_MEMORY_HBM)
                 .Malloc(allocate_size, reinterpret_cast<uint8_t *>(try_reuse_addr), device_id_);
  }
  if (buffer == nullptr) {
    GELOGE(MEMALLOC_FAILED, "[Malloc][Memory] Failed, device_id = %u, size = %zu",
           device_id_, allocate_size);
    REPORT_CALL_ERROR("E19999", "malloc memory failed, device_id = %u, size = %zu",
                      device_id_, allocate_size);
    return nullptr;
  }

  GELOGI("Allocating buffer of size %zu successfully. device_id = %u, address = %p", allocate_size, device_id_, buffer);
  return buffer;
}

void NpuMemoryAllocator::Deallocate(void *data, MemStorageType mem_type) {
  GELOGI("To deallocating buffer, addr = %p", data);
  if (data != nullptr) {
    GELOGI("Deallocating buffer successfully. addr = %p", data);
    if (mem_type == RDMA_HBM) {
      MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).Free(reinterpret_cast<uint8_t *>(data), device_id_);
    } else if (mem_type == HOST_DDR) {
      MemManager::Instance().HostMemInstance(RT_MEMORY_HBM).Free(data);
    } else {
      MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Free(reinterpret_cast<uint8_t *>(data), device_id_);
    }
  }
}

NpuMemoryAllocator *NpuMemoryAllocator::GetAllocator(uint32_t device_id) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = allocators_.find(device_id);
  if (it == allocators_.end()) {
    auto allocator = std::unique_ptr<NpuMemoryAllocator>(new (std::nothrow) NpuMemoryAllocator(device_id));
    if (allocator == nullptr) {
      return nullptr;
    }

    allocators_.emplace(device_id, std::move(allocator));
  }

  return allocators_[device_id].get();
}

void NpuMemoryAllocator::DestroyAllocator() {
  std::lock_guard<std::mutex> lk(mu_);
  int device_id = 0;
  allocators_.erase(device_id);
}
}  // namespace hybrid
}  // namespace ge
