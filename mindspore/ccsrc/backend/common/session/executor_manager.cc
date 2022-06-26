/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "backend/common/session/executor_manager.h"
#include "include/common/thread_pool.h"
namespace mindspore {
namespace session {
ExecutorManager &ExecutorManager::Instance() {
  static ExecutorManager instance{};
  return instance;
}

std::shared_ptr<Executor> ExecutorManager::GetExecutor(const std::string &device_name, uint32_t device_id) {
  std::string device_key = device_name + "_" + std::to_string(device_id);
  auto iter = executors_.find(device_key);
  if (iter != executors_.end()) {
    return iter->second;
  }
  auto executor = std::make_shared<Executor>(device_name, device_id);
  executors_[device_key] = executor;
  return executor;
}

void ExecutorManager::OnEvent(const ExecutorEvent &event) {
  for (auto &item : executors_) {
    auto &executor = item.second;
    if (executor != nullptr) {
      executor->OnEvent(event);
    }
  }
}

void ExecutorManager::Clear() {
  OnEvent(ExecutorEvent::kClear);
  executors_.clear();
  common::ThreadPool::GetInstance().ClearThreadPool();
}
}  // namespace session
}  // namespace mindspore
