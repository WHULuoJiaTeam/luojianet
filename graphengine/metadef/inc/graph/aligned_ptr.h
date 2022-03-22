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

#ifndef GE_ALIGNED_PTR_H_
#define GE_ALIGNED_PTR_H_

#include <memory>
#include <functional>

namespace ge {
class AlignedPtr {
 public:
  using Deleter = std::function<void(uint8_t *)>;
  using Allocator = std::function<void(std::unique_ptr<uint8_t[], Deleter> &base_addr)>;
  explicit AlignedPtr(const size_t buffer_size, const size_t alignment = 16U);
  AlignedPtr() = default;
  ~AlignedPtr() = default;
  AlignedPtr(const AlignedPtr &) = delete;
  AlignedPtr(AlignedPtr &&) = delete;
  AlignedPtr &operator=(const AlignedPtr &) = delete;
  AlignedPtr &operator=(AlignedPtr &&) = delete;

  const uint8_t *Get() const { return aligned_addr_; }
  uint8_t *MutableGet() const { return aligned_addr_; }
  std::unique_ptr<uint8_t[], AlignedPtr::Deleter> Reset();

  static std::shared_ptr<AlignedPtr> BuildFromAllocFunc(const AlignedPtr::Allocator &alloc_func,
                                                        const AlignedPtr::Deleter &delete_func);
  static std::shared_ptr<AlignedPtr> BuildFromData(uint8_t * const data,
                                                   const AlignedPtr::Deleter &delete_func);  /*lint !e148*/
 private:
  std::unique_ptr<uint8_t[], AlignedPtr::Deleter> base_ = nullptr;
  uint8_t *aligned_addr_ = nullptr;
};
}  // namespace ge
#endif//GE_ALIGNED_PTR_H_
