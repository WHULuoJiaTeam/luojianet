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

#include "hybrid/executor/node_done_manager.h"
#include <chrono>
#include "framework/common/debug/ge_log.h"

namespace ge {
namespace hybrid {
namespace {
constexpr int kDefaultWaitTimeoutInSec = 600;
}
bool NodeDoneManager::Cond::Await() {
  std::unique_lock<std::mutex> lk(cond_mu_);
  if (!cv_.wait_for(lk,
                    std::chrono::seconds(kDefaultWaitTimeoutInSec),
                    [&]() { return is_released_ || is_cancelled_; })) {
    GELOGE(INTERNAL_ERROR, "[Invoke][wait_for]Wait timed out.");
    REPORT_INNER_ERROR("E19999", "wait timed out[%d].", kDefaultWaitTimeoutInSec);
    return false;
  }

  return is_released_;
}

void NodeDoneManager::Cond::Reset() {
  std::unique_lock<std::mutex> lk(cond_mu_);
  if (!is_released_ && !is_cancelled_) {
    GELOGW("Called before done, released: %d, cancelled: %d", is_released_, is_cancelled_);
  }

  is_released_ = false;
  is_cancelled_ = false;
}

void NodeDoneManager::Cond::Release() {
  std::unique_lock<std::mutex> lk(cond_mu_);
  is_released_ = true;
  cv_.notify_all();
}

void NodeDoneManager::Cond::Cancel() {
  std::unique_lock<std::mutex> lk(cond_mu_);
  is_cancelled_ = true;
  cv_.notify_all();
}

bool NodeDoneManager::Cond::IsRelease() {
  std::unique_lock<std::mutex> lk(cond_mu_);
  return is_released_;
}

NodeDoneManager::Cond *NodeDoneManager::GetSubject(const NodePtr &node) {
  std::lock_guard<std::mutex> lk(mu_);
  if (destroyed_) {
    GELOGD("Already destroyed.");
    return nullptr;
  }

  auto it = subjects_.find(node);
  if (it == subjects_.end()) {
    return &subjects_[node];
  }

  return &it->second;
}

void NodeDoneManager::Destroy() {
  GELOGD("Start to reset NodeDoneManager.");
  std::lock_guard<std::mutex> lk(mu_);
  GELOGD("Cond size = %zu.", subjects_.size());
  for (auto &sub : subjects_) {
    if (!sub.second.IsRelease()) {
      sub.second.Cancel();
      GELOGD("[%s] Node canceled.", sub.first->GetName().c_str());
    }
  }

  subjects_.clear();
  destroyed_ = true;
  GELOGD("Done resetting NodeDoneManager successfully.");
}

void NodeDoneManager::NodeDone(const NodePtr &node) {
  auto sub = GetSubject(node);
  if (sub != nullptr) {
    sub->Release();
    GELOGD("[%s] Node released.", node->GetName().c_str());
  }
}

bool NodeDoneManager::Await(const NodePtr &node) {
  auto sub = GetSubject(node);
  if (sub == nullptr) {
    return false;
  }

  GELOGD("[%s] Await start. is_released = %s", node->GetName().c_str(), sub->IsRelease() ? "true" : "false");
  bool ret = sub->Await();
  GELOGD("[%s] Await ended. is_released = %s", node->GetName().c_str(), sub->IsRelease() ? "true" : "false");
  return ret;
}

void NodeDoneManager::Reset(const NodePtr &node) {
  auto sub = GetSubject(node);
  if (sub != nullptr) {
    sub->Reset();
    GELOGD("[%s] Node reset.", node->GetName().c_str());
  }
}

void NodeDoneManager::Reset() {
  subjects_.clear();
  destroyed_ = false;
}
}  // namespace hybrid
}  // namespace ge
