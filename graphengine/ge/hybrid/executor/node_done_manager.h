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

#ifndef GE_HYBRID_EXECUTOR_NODE_DONE_COND_MANAGER_H_
#define GE_HYBRID_EXECUTOR_NODE_DONE_COND_MANAGER_H_

#include <condition_variable>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "graph/node.h"

namespace ge {
namespace hybrid {
class NodeDoneManager {
 public:
  void NodeDone(const NodePtr &node);

  bool Await(const NodePtr &node);

  void Reset(const NodePtr &node);

  void Destroy();

  void Reset();

 private:
  class Cond {
   public:
    bool IsRelease();
    void Release();
    void Cancel();
    bool Await();
    void Reset();
   private:
    std::mutex cond_mu_;
    std::condition_variable cv_;
    bool is_released_ = false;
    bool is_cancelled_ = false;
  };

  Cond *GetSubject(const NodePtr &node);
  std::mutex mu_;
  std::unordered_map<NodePtr, Cond> subjects_;
  bool destroyed_ = false;
};
}  // namespace hybrid
}  // namespace ge

#endif // GE_HYBRID_EXECUTOR_NODE_DONE_COND_MANAGER_H_
