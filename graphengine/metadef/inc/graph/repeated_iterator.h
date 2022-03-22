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

#ifndef METADEF_CXX_REPEATED_ITERATOR_H
#define METADEF_CXX_REPEATED_ITERATOR_H
#include <iterator>
#include <cstddef>

namespace ge {
template<typename T>
class RepeatedIterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using pointer = T *;
  using reference = T &;
  using size_type = size_t;

  RepeatedIterator(size_type index, reference value) : index_(index), value_(value) {}

  reference operator*() const {
    return value_;
  }

  pointer operator->() const {
    return &value_;
  }

  RepeatedIterator &operator++() {
    ++index_;
    return *this;
  }
  RepeatedIterator operator++(int) {
    RepeatedIterator ret = *this;
    ++*this;
    return ret;
  }

  friend bool operator==(const RepeatedIterator &lhs, const RepeatedIterator &rhs){
      return (lhs.index_ == rhs.index_) && (&lhs.value_ == &rhs.value_);
  }
  friend bool operator!=(const RepeatedIterator &lhs, const RepeatedIterator &rhs) {
    return !(lhs == rhs);
  };

 private:
  size_type index_;
  reference value_;
};
}  // namespace ge
#endif  //METADEF_CXX_REPEATED_ITERATOR_H
