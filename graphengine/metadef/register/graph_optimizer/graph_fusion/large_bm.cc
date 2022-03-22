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
#include "common/large_bm.h"
#include "graph/debug/ge_log.h"

namespace ge {
constexpr size_t kBitsEachValue = 64;

LargeBitmap::LargeBitmap(size_t size)
    : size_(size), bits_((size + kBitsEachValue - 1) / kBitsEachValue, 0) {}

bool LargeBitmap::operator==(const LargeBitmap &another_bm) const {
  return bits_ == another_bm.bits_;
}

bool LargeBitmap::operator!=(const LargeBitmap &another_bm) const {
  return bits_ != another_bm.bits_;
}

void LargeBitmap::SetValues(uint64_t value) {
  std::fill(bits_.begin(), bits_.end(), value);
}

void LargeBitmap::SetBit(size_t index) {
  if (index >= 0 && index < size_) {
    bits_[index / kBitsEachValue] |= 1ull << (index % kBitsEachValue);
  } else {
    GE_LOGE("index %u is not valid. Total size is %u", index, size_);
    return;
  }
}

bool LargeBitmap::GetBit(size_t index) const {
  if (index >= 0 && index < size_) {
    return bits_[index / kBitsEachValue] & (1ull << (index % kBitsEachValue));
  } else {
    GE_LOGE("index %u is not valid. Total size is %u", index, size_);
    return false;
  }
}

void LargeBitmap::Or(const LargeBitmap &another_bm) {
  size_t index = 0;
  size_t another_size = another_bm.bits_.size();
  for (auto &bit : bits_) {
    if (index >= another_size) {
      return;
    }
    bit |= another_bm.bits_[index];
    ++index;
  }
}

void LargeBitmap::And(const LargeBitmap &another_bm) {
  size_t index = 0;
  size_t another_size = another_bm.bits_.size();
  for (auto &bit : bits_) {
    if (index >= another_size) {
      return;
    }
    bit &= another_bm.bits_[index];
    ++index;
  }
}
}
