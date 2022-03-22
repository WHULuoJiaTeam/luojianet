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

#ifndef GE_HYBRID_COMMON_TENSOR_VALUE_H_
#define GE_HYBRID_COMMON_TENSOR_VALUE_H_

#include <atomic>
#include <cstddef>
#include <memory>
#include "framework/memory/memory_api.h"
#include "framework/common/util.h"

namespace ge {
namespace hybrid {
class NpuMemoryAllocator;
class AllocationAttr;

class TensorBuffer {
 public:
  static std::unique_ptr<TensorBuffer> Create(NpuMemoryAllocator *allocator,
                                              size_t size,
                                              AllocationAttr *attr = nullptr);

  static std::unique_ptr<TensorBuffer> Create(void *buffer, size_t size);

  TensorBuffer(const TensorBuffer &) = delete;
  TensorBuffer &operator = (const TensorBuffer &) = delete;
  ~TensorBuffer();

  void* Release() {
    auto ret = buffer_;
    buffer_ = nullptr;
    return ret;
  }

  void *GetData() {
    return buffer_;
  }

  size_t GetSize() const {
    return size_;
  }

  MemStorageType GetMemType() const {
    return mem_type_;
  }

 private:
  TensorBuffer(NpuMemoryAllocator *allocator, void *buffer, size_t size, MemStorageType mem_type = HBM);

  NpuMemoryAllocator *allocator_ = nullptr;
  void *buffer_ = nullptr;
  size_t size_ = 0;
  MemStorageType mem_type_;
};

class TensorValue {
 public:
  TensorValue() = default;

  explicit TensorValue(std::shared_ptr<TensorBuffer> buffer);

  TensorValue(void *buffer, size_t size);

  ~TensorValue();

  void Destroy();

  void *Release() {
    return buffer_->Release();
  }

  bool IsEmpty() {
    return ref_buffer_ == nullptr && buffer_ == nullptr;
  }

  const void *GetData() const;

  std::string DebugString() const;

  void SetName(const std::string &name) {
    name_ = name;
  }
  
  Status GetMemType(MemStorageType &mem_type) const {
    GE_CHECK_NOTNULL(buffer_);
    return buffer_->GetMemType();
  }

  void *MutableData();

  size_t GetSize() const;

  template<typename T>
  Status CopyScalarValueToHost(T &value) const {
    GE_CHECK_GE(this->GetSize(), sizeof(value));
    return rtMemcpy(&value, sizeof(value), this->GetData(), sizeof(value), RT_MEMCPY_DEVICE_TO_HOST);
  }

 private:
  std::shared_ptr<TensorBuffer> buffer_;
  std::string name_;
  // for weights and variables
  void *ref_buffer_ = nullptr;
  size_t ref_size_ = 0;
  // shape
};
}  // namespace hybrid
}  // namespace ge
#endif  // GE_HYBRID_COMMON_TENSOR_VALUE_H_
