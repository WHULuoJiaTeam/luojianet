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

#ifndef GRAPH_COMPILE_CACHE_POLICY_CACHE_DESC_H_
#define GRAPH_COMPILE_CACHE_POLICY_CACHE_DESC_H_
#include "graph/small_vector.h"
#include "graph/ascend_limits.h"
#include "graph/types.h"
#include "graph/debug/ge_log.h"
#include "hash_utils.h"
#include <securec.h>
#include <string>
#include <vector>
#include <unordered_map>
namespace ge {
using ShapeType = std::vector<int64_t>;
using ShapeRangeType = std::vector<std::pair<int64_t, int64_t>>;

class BinaryHolder {
public:
  BinaryHolder() = default;

  BinaryHolder(const BinaryHolder &other) {
    if (other.GetDataPtr() != nullptr && other.GetDataLen() != 0) {
      data_len_ = other.GetDataLen();
      uint8_t *data = new (std::nothrow) uint8_t[data_len_];
      auto mem_ret = memcpy_s(data, data_len_, other.GetDataPtr(), data_len_);
      holder_.reset(data);
      if (mem_ret != EOK) {
        GELOGE(ge::GRAPH_FAILED, "[BinaryHolder] Memcpy Falied.");
      }
    }
  }
  BinaryHolder &operator=(const BinaryHolder &other) = delete;

  ~BinaryHolder() = default;

  void SharedFrom(void *data, const size_t data_len) {
    data_ptr_ = data;
    data_len_ = data_len;
  }

  const void *GetDataPtr() const noexcept {
    if (holder_.get() != nullptr) {
      return holder_.get();
    }
    return data_ptr_;
  }

  const size_t &GetDataLen() const noexcept {
    return data_len_;
  }

  bool operator!=(const BinaryHolder &second) const {
    if (this->GetDataLen() != second.GetDataLen()) {
      return false;
    }
    auto this_data = this->GetDataPtr();
    auto second_data = second.GetDataPtr();
    if (((this_data == nullptr) && (second_data != nullptr)) ||
        ((this_data != nullptr) && (second_data == nullptr))) {
      return true;
    }
    if ((this_data == nullptr) && (second_data == nullptr)) {
      return false;
    }
    if (memcmp(this_data, second_data, this->GetDataLen()) != 0) {
      return true;
    }
    return false;
  }
private:
  std::unique_ptr<uint8_t[]> holder_ = nullptr;
  void *data_ptr_ = nullptr;
  size_t data_len_ = 0;
};

class CompileCacheDesc {
  friend class CompileCacheHasher;
public:
  CompileCacheDesc() = delete;

  CompileCacheDesc(int64_t unique_id,
                   SmallVector<ShapeType, kDefaultMaxInputNum> shapes,
                   SmallVector<ShapeType, kDefaultMaxInputNum> origin_shapes,
                   SmallVector<ShapeRangeType, kDefaultMaxInputNum> shape_ranges,
                   SmallVector<Format, kDefaultMaxInputNum> formats,
                   SmallVector<Format, kDefaultMaxInputNum> origin_formats,
                   SmallVector<DataType, kDefaultMaxInputNum> data_types,
                   SmallVector<BinaryHolder, kDefaultMaxInputNum> other_desc) {}

  ~CompileCacheDesc() = default;
  static bool IsSameCompileDesc(const CompileCacheDesc &first, const CompileCacheDesc &second) {
    if ((first.unique_id_ != second.unique_id_) ||
        (first.origin_shapes_ != second.origin_shapes_) ||
        (first.formats_ != second.formats_) ||
        (first.origin_formats_ != second.origin_formats_) ||
        (first.data_types_ != second.data_types_) ||
        (first.other_desc_.size() != second.other_desc_.size())) {
      return false;
    }

    for (size_t idx = 0L; idx < first.other_desc_.size(); idx++) {
      if (first.other_desc_[idx] != second.other_desc_[idx]) {
        return false;
      }
    }
    return true;
  }

private:
  int64_t unique_id_ = 0UL;
  SmallVector<ShapeType, kDefaultMaxInputNum> shapes_;
  SmallVector<ShapeType, kDefaultMaxInputNum> origin_shapes_;
  SmallVector<ShapeRangeType, kDefaultMaxInputNum> shape_ranges_;
  SmallVector<Format, kDefaultMaxInputNum> formats_;
  SmallVector<Format, kDefaultMaxInputNum> origin_formats_;
  SmallVector<DataType, kDefaultMaxInputNum> data_types_;
  SmallVector<BinaryHolder, kDefaultMaxInputNum> other_desc_;
};
}  // namespace ge
namespace std {
template<>
struct hash<ge::BinaryHolder> {
  size_t operator()(const ge::BinaryHolder &value) const {
    GE_CHECK_NOTNULL(value.GetDataPtr());
    size_t seed = ge::HashUtils::MultiHash();
    const uint8_t *u8_data = reinterpret_cast<const uint8_t *>(value.GetDataPtr());
    for (size_t idx = 0UL; idx < value.GetDataLen(); idx++) {
      seed = ge::HashUtils::HashCombine(seed, *(u8_data + idx));
    }
    return seed;
  }
};
}  // namespace std
#endif
