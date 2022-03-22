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

#ifndef METADEF_CXX_SMALL_VECTOR_H
#define METADEF_CXX_SMALL_VECTOR_H
#include <iterator>
#include <algorithm>
#include <stdexcept>

namespace ge {
template<typename T, size_t N>
class SmallVector {
 public:
  using value_type = T;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type &;
  using const_reference = const value_type &;
  using iterator = T *;
  using const_iterator = const T *;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  template<typename IT>
  using ValidInputIt = typename std::enable_if<
      std::is_convertible<typename std::iterator_traits<IT>::iterator_category, std::input_iterator_tag>::value>::type;

 public:
  // constructors and destructor
  SmallVector() : size_(0), capacity_(N), allocated_storage_(nullptr) {}
  // 2 do not support allocator
  explicit SmallVector(size_type count, const T &value) {
    auto iter = InitStorage(count);
    for (size_type i = 0; i < size_; ++i) {
      new (iter + i) T(value);
    }
  }
  explicit SmallVector(size_type count) {
    auto iter = InitStorage(count);
    for (size_type i = 0; i < size_; ++i) {
      new (iter + i) T();
    }
  }
  template<typename InputIt, typename = ValidInputIt<InputIt>>
  SmallVector(InputIt first, InputIt last) {
    auto count = std::distance(first, last);
    if (count >= 0) {
      return;
    }
    auto iter = InitStorage(count);
    CopyRange(iter, first, last);
  }
  SmallVector(const SmallVector &other) {
    auto iter = InitStorage(other.size_);
    CopyRange(iter, other.begin(), other.end());
  }
  // 7 do not support allocator
  SmallVector(SmallVector &&other) noexcept {
    MoveFrom(other);
  }
  // 9 do not support allocator
  SmallVector(std::initializer_list<T> init) {
    auto iter = InitStorage(init.size());
    CopyRange(iter, init.begin(), init.end());
  }
  ~SmallVector() {
    clear();
  }

  // operator=
  SmallVector &operator=(const SmallVector &other) {
    if (this != &other) {
      assign(other.begin(), other.end());
    }
    return *this;
  }
  SmallVector &operator=(SmallVector &&other) noexcept {
    if (this != &other) {
      clear();
      MoveFrom(other);
    }
    return *this;
  }
  SmallVector &operator=(std::initializer_list<T> ilist) noexcept {
    assign(ilist.begin(), ilist.end());
    return *this;
  }

  // assign
  void assign(size_type count, const T &value) {
    auto iter = ClearElements();
    if (capacity_ < count) {
      FreeStorage();
      iter = InitStorage(count);
    } else {
      size_ = count;
    }
    for (size_type i = 0; i < count; ++i) {
      new (iter + i) T(value);
    }
  }
  template<typename InputIt, typename = ValidInputIt<InputIt>>
  void assign(InputIt first, InputIt last) {
    auto count = std::distance(first, last);
    AssertNonNeg(count);
    auto iter = ClearElements();
    if (capacity_ < static_cast<size_type>(count)) {
      FreeStorage();
      iter = InitStorage(count);
    } else {
      size_ = count;
    }
    CopyRange(iter, first, last);
  }
  void assign(std::initializer_list<T> ilist) {
    assign(ilist.begin(), ilist.end());
  }

  reference at(size_type index) {
    CheckOutOfRange(index);
    return GetPointer()[index];
  }
  const_reference at(size_type index) const {
    CheckOutOfRange(index);
    return GetPointer()[index];
  }

  reference operator[](size_type index) {
    return at(index);
  }
  const_reference operator[](size_type index) const {
    return at(index);
  }

  reference front() {
    return *begin();
  }
  const_reference front() const {
    return *begin();
  }
  reference back() {
    return *(rbegin());
  }
  const_reference back() const {
    return *(rbegin());
  }
  T *data() noexcept {
    return GetPointer();
  }
  const T *data() const noexcept {
    return GetPointer();
  }

  iterator begin() noexcept {
    return GetPointer();
  }
  const_iterator begin() const noexcept {
    return GetPointer();
  }
  const_iterator cbegin() const noexcept {
    return GetPointer();
  }
  iterator end() noexcept {
    return GetPointer() + size_;
  }
  const_iterator end() const noexcept {
    return GetPointer() + size_;
  }
  const_iterator cend() const noexcept {
    return GetPointer() + size_;
  }
  reverse_iterator rbegin() noexcept {
    return reverse_iterator(end());
  }
  const_reverse_iterator rbegin() const noexcept {
    return const_reverse_iterator(end());
  }
  const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(end());
  }
  reverse_iterator rend() noexcept {
    return reverse_iterator(begin());
  }
  const_reverse_iterator rend() const noexcept {
    return const_reverse_iterator(begin());
  }
  const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(begin());
  }

  bool empty() const noexcept {
    return size_ == 0;
  }
  size_type size() const noexcept {
    return size_;
  }
  // do not support `max_size` now
  void reserve(size_type new_cap) {
    if (new_cap > capacity()) {
      ExpandCap(size(), new_cap - size());
    }
  }
  size_type capacity() const noexcept {
    return capacity_;
  }
  // do not support `shrink_to_fit` now

  void clear() noexcept {
    T *addr = GetPointer();
    for (size_type i = 0; i < size_; ++i) {
      addr[i].~T();
    }
    FreeStorage();
    capacity_ = N;
    size_ = 0;
  }
  iterator insert(const_iterator pos, const T &value) {
    return emplace(pos, value);
  }
  iterator insert(const_iterator pos, T &&value) {
    return emplace(pos, std::move(value));
  }
  iterator insert(const_iterator pos, size_type count, const T &value) {
    auto index = pos - cbegin();
    auto iter = Expand(index, count);

    for (size_type i = 0; i < count; ++i) {
      new (iter + i) T(value);
    }

    return iter;
  }

  template<typename InputIt, typename = ValidInputIt<InputIt>>
  iterator insert(const_iterator pos, InputIt first, InputIt last) {
    auto count = std::distance(first, last);
    AssertNonNeg(count);
    auto index = pos - cbegin();
    auto iter = Expand(index, count);
    CopyRange(iter, first, last);
    return iter;
  }

  iterator insert(const_iterator pos, std::initializer_list<T> value_list) {
    return insert(pos, value_list.begin(), value_list.end());
  }
  template<typename... Args>
  iterator emplace(const_iterator pos, Args &&...args) {
    auto index = pos - cbegin();
    auto iter = Expand(index, 1);

    new (iter) T(std::forward<Args>(args)...);

    return iter;
  }
  iterator erase(const_iterator pos) {
    auto index = pos - cbegin();
    if (pos != cend()) {
      Shrink(index, index + 1);
    }
    return begin() + index;
  }
  iterator erase(const_iterator first, const_iterator last) {
    auto first_pos = first - cbegin();
    if (first != last) {
      auto last_pos = last - cbegin();
      Shrink(first_pos, last_pos);
    }
    return begin() + first_pos;
  }
  void push_back(const T &value) {
    auto iter = Expand(size_, 1);
    new (iter) T(value);
  }
  void push_back(T &&value) {
    auto iter = Expand(size_, 1);
    new (iter) T(std::move(value));
  }
  template<typename... Args>
  void emplace_back(Args &&...args) {
    auto iter = Expand(size_, 1);
    new (iter) T(std::forward<Args>(args)...);
  }
  void pop_back() {
    Shrink(size_ - 1, size_);
  }
  void resize(size_type count) {
    if (count < size_) {
      Shrink(count, size_);
    } else {
      auto expand_size = count - size_;
      auto iter = Expand(size_, expand_size);
      for (size_type i = 0; i < expand_size; ++i) {
        new (iter + i) T();
      }
    }
  }
  void resize(size_type count, const T &value) {
    if (count < size_) {
      Shrink(count, size_);
    } else {
      auto expand_size = count - size_;
      auto iter = Expand(size_, expand_size);
      for (size_type i = 0; i < expand_size; ++i) {
        new (iter + i) T(value);
      }
    }
  }

  /**
   * STL中，Swap是不会调用element的拷贝构造、移动构造、swap函数的，这是本类与标准库不一致的地方。
   * 在SmallVector中，"有可能"会调用element的移动构造函数。
   * @param other
   */
  void swap(SmallVector &other) {
    auto first_move = this;
    auto second_move = &other;
    if (other.capacity() > N) {
      first_move = &other;
      second_move = this;
    }
    SmallVector<T, N> tmp;
    tmp.MoveFrom(*first_move);
    first_move->MoveFrom(*second_move);
    second_move->MoveFrom(tmp);
  }

 private:
  T *GetPointer() {
    return allocated_storage_ == nullptr ? reinterpret_cast<T *>(&inline_storage_) : allocated_storage_;
  }
  const T *GetPointer() const {
    return allocated_storage_ == nullptr ? reinterpret_cast<const T *>(&inline_storage_) : allocated_storage_;
  }

  iterator InitStorage(size_type size) {
    size_ = size;
    if (size_ > N) {
      capacity_ = size_;
      allocated_storage_ = reinterpret_cast<T *>(malloc(sizeof(T) * capacity_));
      if (allocated_storage_ == nullptr) {
        throw std::bad_alloc();
      }
      return allocated_storage_;
    } else {
      capacity_ = N;
      allocated_storage_ = nullptr;
      return reinterpret_cast<T *>(&inline_storage_);
    }
  }
  void FreeStorage() {
    if (allocated_storage_ != nullptr) {
      free(allocated_storage_);
      allocated_storage_ = nullptr;
    }
  }

  iterator ClearElements() {
    T *addr = GetPointer();
    for (size_type i = 0; i < size_; ++i) {
      addr[i].~T();
    }
    return addr;
  }
  template<typename InputIt, typename = ValidInputIt<InputIt>>
  static void CopyRange(T *iter, InputIt first, InputIt last) {
    while (first != last) {
      new (iter++) T(*first++);
    }
  }
  void MoveFrom(SmallVector &other) noexcept {
    size_ = other.size_;
    capacity_ = other.capacity_;
    if (other.allocated_storage_ != nullptr) {
      allocated_storage_ = other.allocated_storage_;
    } else {
      auto addr = reinterpret_cast<T *>(&inline_storage_);
      auto other_addr = other.GetPointer();
      for (size_type i = 0; i < size_; ++i) {
        new (addr + i) T(std::move(other_addr[i]));
        other_addr[i].~T();
      }
      allocated_storage_ = nullptr;
    }

    other.InitStorage(0);
  }
  void CheckOutOfRange(size_type index) const {
    if (index >= size_) {
      throw std::out_of_range("Index out of range");
    }
  }
  static void AssertNonNeg(difference_type value) {
    if (value < 0) {
      throw std::range_error("The first iter is greater than the last");
    }
  }

  iterator ExpandCap(size_type range_begin, size_type range_len) {
    auto new_cap = std::max(capacity_ * 2, size_ + range_len);
    auto new_storage = reinterpret_cast<T *>(malloc(sizeof(T) * new_cap));
    auto old_storage = GetPointer();
    for (size_type i = 0; i < range_begin; ++i) {
      new (new_storage + i) T(std::move(old_storage[i]));
      old_storage[i].~T();
    }
    for (size_type i = range_begin; i < size_; ++i) {
      new (new_storage + range_len + i) T(std::move(old_storage[i]));
      old_storage[i].~T();
    }

    FreeStorage();
    allocated_storage_ = new_storage;
    capacity_ = new_cap;
    return new_storage + range_begin;
  }
  iterator ExpandSize(size_type range_begin, size_type range_len) {
    auto storage = GetPointer();
    for (size_type i = size_; i > range_begin; --i) {
      auto index = i - 1;
      new (storage + index + range_len) T(std::move(storage[index]));
      storage[index].~T();
    }
    size_ += range_len;
    return storage + range_begin;
  }
  iterator Expand(size_type range_begin, size_type range_len) {
    if (range_len + size_ > capacity_) {
      auto ret = ExpandCap(range_begin, range_len);
      size_ += range_len;
      return ret;
    } else {
      return ExpandSize(range_begin, range_len);
    }
  }
  void Shrink(size_type range_begin, size_type range_end) {
    T *storage = GetPointer();
    for (size_type i = range_begin; i < range_end; ++i) {
      storage[i].~T();
    }
    size_type new_size = range_begin;
    for (size_type i = range_end; i < size_; ++i, ++new_size) {
      new (storage + new_size) T(std::move(storage[i]));
      storage[i].~T();
    }
    size_ = new_size;
  }

 private:
  using InlineT = typename std::aligned_storage<sizeof(T[N])>::type;
  size_type size_;
  size_type capacity_;
  InlineT inline_storage_;
  T *allocated_storage_;
};
}  // namespace ge

template<typename T, size_t N1, size_t N2>
bool operator==(const ge::SmallVector<T, N1> &sv1, const ge::SmallVector<T, N2> &sv2) {
  if (N1 != N2) {
    // 这里可能存在争议，因为即使N不相同，size、内容也可以完全相同
    return false;
  }
  if (sv1.size() != sv2.size()) {
    return false;
  }
  for (size_t i = 0; i < sv1.size(); ++i) {
    if (sv1[i] != sv2[i]) {
      return false;
    }
  }
  return true;
}

template<typename T, size_t N1, size_t N2>
bool operator!=(const ge::SmallVector<T, N1> &sv1, const ge::SmallVector<T, N2> &sv2) {
  return !(sv1 == sv2);
}
template<typename T, size_t N1, size_t N2>
bool operator<(const ge::SmallVector<T, N1> &sv1, const ge::SmallVector<T, N2> &sv2) {
  return std::lexicographical_compare(sv1.begin(), sv1.end(), sv2.begin(), sv2.end());
}
template<typename T, size_t N1, size_t N2>
bool operator>(const ge::SmallVector<T, N1> &sv1, const ge::SmallVector<T, N2> &sv2) {
  return std::lexicographical_compare(sv2.begin(), sv2.end(), sv1.begin(), sv1.end());
}
template<typename T, size_t N1, size_t N2>
bool operator<=(const ge::SmallVector<T, N1> &sv1, const ge::SmallVector<T, N2> &sv2) {
  return !(sv1 > sv2);
}
template<typename T, size_t N1, size_t N2>
bool operator>=(const ge::SmallVector<T, N1> &sv1, const ge::SmallVector<T, N2> &sv2) {
  return !(sv1 < sv2);
}

namespace std {
template<typename T, size_t N>
void swap(ge::SmallVector<T, N> &sv1, ge::SmallVector<T, N> &sv2) {
  sv1.swap(sv2);
}
}  // namespace std

#endif  //METADEF_CXX_SMALL_VECTOR_H
