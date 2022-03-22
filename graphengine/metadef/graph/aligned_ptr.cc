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

#include "graph/aligned_ptr.h"
#include "graph/utils/mem_utils.h"
#include "graph/debug/ge_log.h"

namespace ge {
AlignedPtr::AlignedPtr(const size_t buffer_size, const size_t alignment) {
  size_t alloc_size = buffer_size;
  if (alignment > 0U) {
    alloc_size = buffer_size + alignment - 1U;
  }
  if ((buffer_size == 0u) || (alloc_size < buffer_size)) {
    GELOGW("[Allocate][Buffer] Allocate empty buffer or overflow, size=%zu, alloc_size=%zu", buffer_size, alloc_size);
    return;
  }

  base_ =
    std::unique_ptr<uint8_t[], AlignedPtr::Deleter>(new (std::nothrow) uint8_t[alloc_size], [](const uint8_t *ptr) {
    delete[] ptr;
    ptr = nullptr;
  });
  if (base_ == nullptr) {
    GELOGW("[Allocate][Buffer] Allocate buffer failed, size=%zu", alloc_size);
    return;
  }

  if (alignment == 0U) {
    aligned_addr_ = base_.get();
  } else {
    const size_t offset = alignment - 1U;
    aligned_addr_ =
        reinterpret_cast<uint8_t *>((static_cast<size_t>(reinterpret_cast<uintptr_t>(base_.get())) + offset) & ~offset);
  }
}

std::unique_ptr<uint8_t[], AlignedPtr::Deleter> AlignedPtr::Reset() {
  const auto deleter_func = base_.get_deleter();
  if (deleter_func == nullptr) {
    base_.release();
    return std::unique_ptr<uint8_t[], AlignedPtr::Deleter>(aligned_addr_, nullptr);
  } else {
    const auto base_addr = base_.release();
    return
      std::unique_ptr<uint8_t[], AlignedPtr::Deleter>(aligned_addr_, [deleter_func, base_addr](const uint8_t *ptr) {
      deleter_func(base_addr);
      ptr = nullptr;
    });
  }
}

std::shared_ptr<AlignedPtr> AlignedPtr::BuildFromAllocFunc(const AlignedPtr::Allocator &alloc_func,
                                                           const AlignedPtr::Deleter &delete_func) {
  if ((alloc_func == nullptr) || (delete_func == nullptr)) {
      REPORT_INNER_ERROR("E19999", "alloc_func or delete_func is nullptr, check invalid");
      GELOGE(FAILED, "[Check][Param] alloc_func/delete_func is null");
      return nullptr;
  }
  const auto aligned_ptr = MakeShared<AlignedPtr>();
  if (aligned_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "create AlignedPtr failed.");
    GELOGE(INTERNAL_ERROR, "[Create][AlignedPtr] make shared for AlignedPtr failed");
    return nullptr;
  }
  aligned_ptr->base_.reset();
  alloc_func(aligned_ptr->base_);
  aligned_ptr->base_.get_deleter() = delete_func;
  if (aligned_ptr->base_ == nullptr) {
    REPORT_CALL_ERROR("E19999", "allocate for AlignedPtr failed");
    GELOGE(FAILED, "[Call][AllocFunc] allocate for AlignedPtr failed");
    return nullptr;
  }
  aligned_ptr->aligned_addr_ = aligned_ptr->base_.get();
  return aligned_ptr;
}

std::shared_ptr<AlignedPtr> AlignedPtr::BuildFromData(uint8_t * const data, const AlignedPtr::Deleter &delete_func) {
  if ((data == nullptr) || (delete_func == nullptr)) {
    REPORT_INNER_ERROR("E19999", "data is nullptr or delete_func is nullptr");
    GELOGE(FAILED, "[Check][Param] data/delete_func is null");
    return nullptr;
  }
  const auto aligned_ptr = MakeShared<AlignedPtr>();
  if (aligned_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "create AlignedPtr failed.");
    GELOGE(INTERNAL_ERROR, "[Create][AlignedPtr] make shared for AlignedPtr failed");
    return nullptr;
  }
  aligned_ptr->base_.reset(data);
  aligned_ptr->base_.get_deleter() = delete_func;
  aligned_ptr->aligned_addr_ = aligned_ptr->base_.get();
  return aligned_ptr;
}
}  // namespace ge
