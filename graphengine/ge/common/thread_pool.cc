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

#include "common/thread_pool.h"

#include <atomic>
#include <functional>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "external/register/register_types.h"

namespace ge {
ThreadPool::ThreadPool(uint32_t size) : is_stoped_(false) {
  idle_thrd_num_ = size < 1 ? 1 : size;

  for (uint32_t i = 0; i < idle_thrd_num_; ++i) {
    pool_.emplace_back(ThreadFunc, this);
  }
}

ThreadPool::~ThreadPool() {
  is_stoped_.store(true);
  {
    std::unique_lock<std::mutex> lock{m_lock_};
    cond_var_.notify_all();
  }

  for (std::thread &thd : pool_) {
    if (thd.joinable()) {
      try {
        thd.join();
      } catch (const std::system_error &) {
        GELOGW("system_error");
      } catch (...) {
        GELOGW("exception");
      }
    }
  }
}

void ThreadPool::ThreadFunc(ThreadPool *thread_pool) {
  if (thread_pool == nullptr) {
    return;
  }
  while (!thread_pool->is_stoped_) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock{thread_pool->m_lock_};
      thread_pool->cond_var_.wait(
        lock, [thread_pool] { return thread_pool->is_stoped_.load() || !thread_pool->tasks_.empty(); });
      if (thread_pool->is_stoped_ && thread_pool->tasks_.empty()) {
        return;
      }
      task = std::move(thread_pool->tasks_.front());
      thread_pool->tasks_.pop();
    }
    --thread_pool->idle_thrd_num_;
    task();
    ++thread_pool->idle_thrd_num_;
  }
}
}  // namespace ge
