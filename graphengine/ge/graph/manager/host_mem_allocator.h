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

#ifndef GE_GRAPH_MANAGER_HOST_MEM_ALLOCATOR_H_
#define GE_GRAPH_MANAGER_HOST_MEM_ALLOCATOR_H_

#include <mutex>
#include <map>

#include "framework/common/ge_inner_error_codes.h"
#include "graph/aligned_ptr.h"
#include "runtime/mem.h"

namespace ge {
class HostMemAllocator {
 public:
  explicit HostMemAllocator(rtMemType_t) {}
  ~HostMemAllocator() = default;

  HostMemAllocator(const HostMemAllocator &) = delete;
  HostMemAllocator &operator=(const HostMemAllocator &) = delete;

  Status Initialize() {
    Clear();
    return SUCCESS;
  }
  void Finalize() { Clear(); }

  const void *Malloc(const std::shared_ptr<AlignedPtr>& aligned_ptr, size_t size);
  uint8_t *Malloc(size_t size);
  Status Free(const void *memory_addr);

  std::pair<size_t, std::shared_ptr<AlignedPtr>> GetAlignedPtr(const void *addr) { return allocated_blocks_[addr]; }

 private:
  void Clear();

  std::map<const void *, std::pair<size_t, std::shared_ptr<AlignedPtr>>> allocated_blocks_;
  // lock around all operations
  mutable std::mutex mutex_;
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_HOST_MEM_ALLOCATOR_H_
