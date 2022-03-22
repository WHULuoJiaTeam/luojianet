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

#ifndef GE_HYBRID_COMMON_MEMORY_ALLOCATOR_H_
#define GE_HYBRID_COMMON_MEMORY_ALLOCATOR_H_

#include <cstdint>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include "external/ge/ge_api_error_codes.h"
#include "framework/memory/memory_api.h"

namespace ge {
namespace hybrid {
class AllocationAttr {
 public:
  AllocationAttr() = default;
  explicit AllocationAttr(int padding);
  explicit AllocationAttr(void *try_reuse_addr);
  AllocationAttr(int padding, void *try_reuse_addr, MemStorageType = HBM);
  ~AllocationAttr() = default;
  void SetMemType(MemStorageType memType) { mem_type_ = memType; }
  MemStorageType GetMemType() { return mem_type_; }

 private:
  friend class NpuMemoryAllocator;
  int padding_ = 0;
  void *try_reuse_addr_ = nullptr;
  MemStorageType mem_type_ = HBM;
};

class NpuMemoryAllocator {
 public:
  ~NpuMemoryAllocator() = default;
  static NpuMemoryAllocator *GetAllocator(uint32_t device_id);
  static NpuMemoryAllocator *GetAllocator();
  static void DestroyAllocator();
  static AllocationAttr* AttrWithDefaultPadding() {
    static AllocationAttr attr(kDefaultPadding, nullptr);
    return &attr;
  }

  void *Allocate(std::size_t size, AllocationAttr *attr = nullptr);
  void Deallocate(void *data, MemStorageType mem_type = HBM);

  static constexpr int kDefaultPadding = 32;
 private:
  explicit NpuMemoryAllocator(uint32_t device_id);
  uint32_t device_id_;

  static std::map<uint32_t, std::unique_ptr<NpuMemoryAllocator>> allocators_;
  static std::mutex mu_;
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_COMMON_MEMORY_ALLOCATOR_H_
