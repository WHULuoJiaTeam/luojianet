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

#include "hybrid/common/tensor_value.h"
#include <sstream>
#include "framework/common/debug/ge_log.h"
#include "hybrid/common/npu_memory_allocator.h"

namespace ge {
namespace hybrid {
TensorBuffer::TensorBuffer(NpuMemoryAllocator *allocator, void *buffer, size_t size, MemStorageType mem_type)
    : allocator_(allocator), buffer_(buffer), size_(size), mem_type_(mem_type) {}

std::unique_ptr<TensorBuffer> TensorBuffer::Create(NpuMemoryAllocator *allocator, size_t size, AllocationAttr *attr) {
  void *buffer = nullptr;
  if (size == 0) {
    GELOGD("size is 0");
    return Create(buffer, 0U);
  }

  if (allocator == nullptr) {
    GELOGE(INTERNAL_ERROR, "[Check][Param:NpuMemoryAllocator] allocator is NULL.");
    REPORT_INNER_ERROR("E19999", "input allocator is NULL.");
    return nullptr;
  }

  MemStorageType mem_type = HBM;
  if (attr != nullptr) {
    mem_type = attr->GetMemType();
  }
  buffer = allocator->Allocate(size, attr);
  if (buffer == nullptr) {
    GELOGE(MEMALLOC_FAILED, "[Allocate][Memory] Failed. size = %zu.", size);
    REPORT_CALL_ERROR("E19999", "allocate failed, size = %zu.", size);
    return nullptr;
  }

  GELOGD("Tensor created. addr = %p, size = %zu, mem_type = %d", buffer, size, static_cast<int32_t>(mem_type));
  return std::unique_ptr<TensorBuffer>(new (std::nothrow) TensorBuffer(allocator, buffer, size, mem_type));
}

std::unique_ptr<TensorBuffer> TensorBuffer::Create(void *buffer, size_t size) {
  GELOGD("Tensor created. addr = %p, size = %zu", buffer, size);
  return std::unique_ptr<TensorBuffer>(new (std::nothrow) TensorBuffer(nullptr, buffer, size));
}

TensorBuffer::~TensorBuffer() {
  if (allocator_ != nullptr) {
    allocator_->Deallocate(buffer_, mem_type_);
    buffer_ = nullptr;
  }
}

TensorValue::TensorValue(std::shared_ptr<TensorBuffer> buffer) : buffer_(std::move(buffer)) {
}

TensorValue::TensorValue(void *buffer, size_t size) : ref_buffer_(buffer), ref_size_(size) {
}

TensorValue::~TensorValue() { Destroy(); }

void TensorValue::Destroy() {
  if (buffer_ != nullptr) {
    GELOGD("Unref tensor: %s", DebugString().c_str());
    buffer_.reset();
  }
}

size_t TensorValue::GetSize() const {
  if (ref_buffer_ != nullptr) {
    return ref_size_;
  }

  if (buffer_ == nullptr) {
    GELOGD("TensorValue[%s] is empty", name_.c_str());
    return 0;
  }

  return buffer_->GetSize();
}

const void *TensorValue::GetData() const {
  if (ref_buffer_ != nullptr) {
    return ref_buffer_;
  }

  if (buffer_ == nullptr) {
    GELOGD("TensorValue[%s] is empty", name_.c_str());
    return nullptr;
  }
  return buffer_->GetData();
}

void *TensorValue::MutableData() {
  if (ref_buffer_ != nullptr) {
    return ref_buffer_;
  }

  if (buffer_ == nullptr) {
    GELOGD("TensorValue[%s] is empty", name_.c_str());
    return nullptr;
  }

  return buffer_->GetData();
}

std::string TensorValue::DebugString() const {
  std::stringstream ss;
  ss << "TensorValue[";
  if (name_.empty()) {
    ss << "unnamed] ";
  } else {
    ss << name_ << "] ";
  }

  if (ref_buffer_ != nullptr) {
    ss << "ref_addr = " << ref_buffer_ << ", size = " << ref_size_;
  } else if (buffer_ != nullptr) {
    ss << "addr = " << buffer_->GetData() << ", size = " << buffer_->GetSize();
  } else {
    ss << "addr = (nil)";
  }

  return ss.str();
}
}  // namespace hybrid
}  // namespace ge
