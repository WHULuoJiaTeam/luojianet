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

#ifndef EXECUTE_GRAPH_OBJECT_POOL_H
#define EXECUTE_GRAPH_OBJECT_POOL_H

#include <memory>
#include <queue>
namespace ge {
constexpr size_t kDefaultPoolSize = 100UL;

template<class T, size_t N = kDefaultPoolSize>
class ObjectPool {
 public:
  ObjectPool() = default;
  ~ObjectPool() = default;
  ObjectPool(ObjectPool &) = delete;
  ObjectPool(ObjectPool &&) = delete;
  ObjectPool &operator=(const ObjectPool &) = delete;

  template<typename... Args>
  std::unique_ptr<T> Acquire(Args &&...args) {
    if (!handlers_.empty()) {
      std::unique_ptr<T> tmp(std::move(handlers_.front()));
      handlers_.pop();
      return std::move(tmp);
    }
    return std::move(std::unique_ptr<T>(new (std::nothrow) T(args...)));
  }

  void Release(std::unique_ptr<T> ptr) {
    if ((handlers_.size() < N) && (ptr != nullptr)) {
      handlers_.push(std::move(ptr));
    }
  }

  bool IsEmpty() const {
    return handlers_.empty();
  }

  bool IsFull() const {
    return handlers_.size() >= N;
  }

 private:
  std::queue<std::unique_ptr<T>> handlers_;
};
}  // namespace ge
#endif  // EXECUTE_GRAPH_OBJECT_POOL_H
