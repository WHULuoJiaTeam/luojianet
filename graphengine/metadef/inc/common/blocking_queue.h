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

#ifndef INC_COMMON_BLOCKING_QUEUE_H_
#define INC_COMMON_BLOCKING_QUEUE_H_

#include <cstdint>
#include <condition_variable>
#include <list>
#include <mutex>

namespace ge {
constexpr uint32_t kDefaultMaxQueueSize = 2048U;

template <typename T>
class BlockingQueue {
 public:
  explicit BlockingQueue(const uint32_t max_size = kDefaultMaxQueueSize) : max_size_(max_size), is_stoped_(false) {}

  ~BlockingQueue() {}

  bool Pop(T &item) {
    std::unique_lock<std::mutex> lock(mutex_);

    while (queue_.empty() && (!is_stoped_)) {
      empty_cond_.wait(lock);
    }

    if (is_stoped_) {
      return false;
    }

    item = std::move(queue_.front());
    queue_.pop_front();

    full_cond_.notify_one();

    return true;
  }

  bool Push(const T &item, const bool is_wait = true) {
    std::unique_lock<std::mutex> lock(mutex_);

    while ((queue_.size() >= max_size_) && (!is_stoped_)) {
      if (!is_wait) {
        return false;
      }
      full_cond_.wait(lock);
    }

    if (is_stoped_) {
      return false;
    }

    queue_.push_back(item);

    empty_cond_.notify_one();

    return true;
  }

  bool Push(T &&item, const bool is_wait = true) {
    std::unique_lock<std::mutex> lock(mutex_);

    while ((queue_.size() >= max_size_) && (!is_stoped_)) {
      if (!is_wait) {
        return false;
      }
      full_cond_.wait(lock);
    }

    if (is_stoped_) {
      return false;
    }

    queue_.emplace_back(std::move(item));

    empty_cond_.notify_one();

    return true;
  }

  void Stop() {
    {
      const std::unique_lock<std::mutex> lock(mutex_);
      is_stoped_ = true;
    }

    full_cond_.notify_all();
    empty_cond_.notify_all();
  }

  void Restart() {
    const std::unique_lock<std::mutex> lock(mutex_);
    is_stoped_ = false;
  }

  // if the queue is stoped ,need call this function to release the unprocessed items
  std::list<T> GetRemainItems() {
    const std::unique_lock<std::mutex> lock(mutex_);

    if (!is_stoped_) {
      return std::list<T>();
    }

    return queue_;
  }

  bool IsFull() {
    const std::unique_lock<std::mutex> lock(mutex_);
    return queue_.size() >= max_size_;
  }

  void Clear() {
    const std::unique_lock<std::mutex> lock(mutex_);
    queue_.clear();
  }

  void SetMaxSize(const uint32_t size) {
    const std::unique_lock<std::mutex> lock(mutex_);
    if (size == 0U) {
      max_size_ = kDefaultMaxQueueSize;
      return;
    }
    max_size_ = size;
  }

  uint32_t Size() {
    const std::unique_lock<std::mutex> lock(mutex_);
    return static_cast<uint32_t>(queue_.size());
  }

 private:
  std::list<T> queue_;
  std::mutex mutex_;
  std::condition_variable empty_cond_;
  std::condition_variable full_cond_;
  uint32_t max_size_;

  bool is_stoped_;
};
}  // namespace ge

#endif  // INC_COMMON_BLOCKING_QUEUE_H_
