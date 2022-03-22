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

#ifndef LUOJIANET_MS_CCSRC_MINDDATA_DATASET_UTIL_MEMORY_POOL_H_
#define LUOJIANET_MS_CCSRC_MINDDATA_DATASET_UTIL_MEMORY_POOL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include "minddata/dataset/util/status.h"

namespace luojianet_ms {
namespace dataset {
// Abstract class of a memory pool
class MemoryPool {
 public:
  // Allocate a block of size n
  virtual Status Allocate(size_t, void **) = 0;

  // Enlarge or shrink a block from oldSz to newSz
  virtual Status Reallocate(void **, size_t old_sz, size_t new_sz) = 0;

  // Free a pointer
  virtual void Deallocate(void *) = 0;

  // What is the maximum size I can allocate ?
  virtual uint64_t get_max_size() const = 0;

  virtual int PercentFree() const = 0;

  // Destructor
  virtual ~MemoryPool() {}
};

Status DeMalloc(std::size_t s, void **p, bool);
}  // namespace dataset
}  // namespace luojianet_ms

void *operator new(std::size_t, luojianet_ms::Status *, std::shared_ptr<luojianet_ms::dataset::MemoryPool>);

void *operator new[](std::size_t, luojianet_ms::Status *, std::shared_ptr<luojianet_ms::dataset::MemoryPool>);

void operator delete(void *, std::shared_ptr<luojianet_ms::dataset::MemoryPool>);

void operator delete[](void *, std::shared_ptr<luojianet_ms::dataset::MemoryPool>);

#endif  // LUOJIANET_MS_CCSRC_MINDDATA_DATASET_UTIL_MEMORY_POOL_H_
