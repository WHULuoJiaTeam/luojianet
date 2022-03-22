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

#ifndef GE_GE_RUNTIME_TASK_TASK_H_
#define GE_GE_RUNTIME_TASK_TASK_H_

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include "runtime/rt_model.h"
#include "ge_runtime/model_context.h"
#include "framework/ge_runtime/task_info.h"
#include "external/runtime/rt_error_codes.h"

namespace ge {
namespace model_runner {
class Task {
 public:
  Task() {}

  virtual ~Task() {}

  virtual bool Distribute() = 0;

  virtual void *Args() { return nullptr; }

  virtual std::string task_name() const { return ""; }
};

template <class T>
class TaskRepeater : public Task {
  static_assert(std::is_base_of<TaskInfo, T>(), "Wrong TaskInfo Type!");

 public:
  TaskRepeater(const ModelContext &model_context, std::shared_ptr<T> task_info) {}

  virtual ~TaskRepeater() {}

  virtual bool Distribute() = 0;
};
}  // namespace model_runner
}  // namespace ge

#endif  // GE_GE_RUNTIME_TASK_TASK_H_
