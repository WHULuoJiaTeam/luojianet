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

#ifndef GE_COMMON_THREAD_POOL_H_
#define GE_COMMON_THREAD_POOL_H_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "external/ge/ge_api_error_codes.h"
#include "external/graph/types.h"
#include "common/ge/ge_util.h"

namespace ge {
using ThreadTask = std::function<void()>;

class ThreadPool {
 public:
  explicit ThreadPool(uint32_t size = 4);
  ~ThreadPool();

  template <class Func, class... Args>
  auto commit(Func &&func, Args &&... args) -> std::future<decltype(func(args...))> {
    GELOGD("commit run task enter.");
    using retType = decltype(func(args...));
    std::future<retType> fail_future;
    if (is_stoped_.load()) {
      GELOGE(ge::FAILED, "thread pool has been stopped.");
      return fail_future;
    }

    auto bindFunc = std::bind(std::forward<Func>(func), std::forward<Args>(args)...);
    auto task = ge::MakeShared<std::packaged_task<retType()>>(bindFunc);
    if (task == nullptr) {
      GELOGE(ge::FAILED, "Make shared failed.");
      return fail_future;
    }
    std::future<retType> future = task->get_future();
    {
      std::lock_guard<std::mutex> lock{m_lock_};
      tasks_.emplace([task]() { (*task)(); });
    }
    cond_var_.notify_one();
    GELOGD("commit run task end");
    return future;
  }

  static void ThreadFunc(ThreadPool *thread_pool);

 private:
  std::vector<std::thread> pool_;
  std::queue<ThreadTask> tasks_;
  std::mutex m_lock_;
  std::condition_variable cond_var_;
  std::atomic<bool> is_stoped_;
  std::atomic<uint32_t> idle_thrd_num_;
};
}  // namespace ge

#endif  // GE_COMMON_THREAD_POOL_H_
