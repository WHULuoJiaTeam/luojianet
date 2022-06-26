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

#ifndef MINDSPORE_CCSRC_PS_SCHEDULER_H_
#define MINDSPORE_CCSRC_PS_SCHEDULER_H_

#include <memory>
#include "ps/core/scheduler_node.h"
#include "ps/core/ps_scheduler_node.h"
#include "ps/util.h"
#include "ps/ps_context.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace ps {
class BACKEND_EXPORT Scheduler {
 public:
  static Scheduler &GetInstance();

  void Run();

 private:
  Scheduler() {
    if (scheduler_node_ == nullptr) {
      bool is_fl_mode = PSContext::instance()->server_mode() == ps::kServerModeFL ||
                        PSContext::instance()->server_mode() == ps::kServerModeHybrid;
      if (is_fl_mode) {
        scheduler_node_ = std::make_unique<core::SchedulerNode>();
      } else {
        scheduler_node_ = std::make_unique<core::PSSchedulerNode>();
      }
    }
  }

  ~Scheduler() = default;
  Scheduler(const Scheduler &) = delete;
  Scheduler &operator=(const Scheduler &) = delete;
  std::unique_ptr<core::SchedulerNode> scheduler_node_;
};
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_SCHEDULER_H_
