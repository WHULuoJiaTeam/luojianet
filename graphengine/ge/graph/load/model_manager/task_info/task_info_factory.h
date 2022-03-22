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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_TASK_INFO_FACTORY_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_TASK_INFO_FACTORY_H_

#include <functional>
#include <map>
#include <memory>
#include <string>

#include "framework/common/debug/ge_log.h"
#include "common/ge/ge_util.h"
#include "runtime/rt_model.h"

namespace ge {
class TaskInfo;
using TaskInfoPtr = std::shared_ptr<TaskInfo>;

class TaskInfoFactory {
 public:
  // TaskManagerCreator function def
  using TaskInfoCreatorFun = std::function<TaskInfoPtr(void)>;

  static TaskInfoFactory &Instance() {
    static TaskInfoFactory instance;
    return instance;
  }

  TaskInfoPtr Create(rtModelTaskType_t task_type) {
    auto iter = creator_map_.find(task_type);
    if (iter == creator_map_.end()) {
      GELOGW("Cannot find task type %d in inner map.", static_cast<int>(task_type));
      return nullptr;
    }

    return iter->second();
  }

  // TaskInfo registerar
  class Registerar {
   public:
    Registerar(rtModelTaskType_t type, const TaskInfoCreatorFun func) {
      TaskInfoFactory::Instance().RegisterCreator(type, func);
    }

    ~Registerar() {}
  };

 private:
  TaskInfoFactory() {}

  ~TaskInfoFactory() {}

  // register creator, this function will call in the constructor
  void RegisterCreator(rtModelTaskType_t type, const TaskInfoCreatorFun func) {
    auto iter = creator_map_.find(type);
    if (iter != creator_map_.end()) {
      GELOGD("TaskManagerFactory::RegisterCreator: %d creator already exist", static_cast<int>(type));
      return;
    }

    creator_map_[type] = func;
  }

  std::map<rtModelTaskType_t, TaskInfoCreatorFun> creator_map_;
};

#define REGISTER_TASK_INFO(type, clazz)      \
  TaskInfoPtr Creator_##type##_Task_Info() { \
    std::shared_ptr<clazz> ptr = nullptr;    \
    ptr = MakeShared<clazz>();               \
    return ptr;                              \
  }                                          \
  TaskInfoFactory::Registerar g_##type##_Task_Info_Creator(type, Creator_##type##_Task_Info);
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_TASK_INFO_FACTORY_H_
