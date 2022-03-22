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

#ifndef GE_GRAPH_MANAGER_BLOCK_MEMORY_H_
#define GE_GRAPH_MANAGER_BLOCK_MEMORY_H_
namespace ge {
struct Block;
typedef bool (*Comparison)(const Block *, const Block *);
using BlockBin = std::set<Block *, Comparison>;

struct Block {
  uint32_t device_id;  // npu device id
  size_t size;         // block size in bytes
  BlockBin *bin;       // owning block bin
  uint8_t *ptr;        // memory address
  bool allocated;      // in-use flag
  Block *prev;         // prev block if split from a larger allocation
  Block *next;         // next block if split from a larger allocation

  Block(uint32_t device, size_t size, BlockBin *bin, uint8_t *ptr)
      : device_id(device), size(size), bin(bin), ptr(ptr), allocated(false), prev(nullptr), next(nullptr) {}

  // constructor for search key
  Block(uint32_t device, size_t size, uint8_t *ptr)
      : device_id(device), size(size), bin(nullptr), ptr(ptr), allocated(false), prev(nullptr), next(nullptr) {}

  bool IsSplit() const { return (prev != nullptr) || (next != nullptr); }
};
}  // namespace ge
#endif  // GE_GRAPH_MANAGER_BLOCK_MEMORY_H_
