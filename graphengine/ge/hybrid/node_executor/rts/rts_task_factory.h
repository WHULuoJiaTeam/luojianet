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

#ifndef GE_HYBRID_NODE_EXECUTOR_RTS_TASK_FACTORY_H_
#define GE_HYBRID_NODE_EXECUTOR_RTS_TASK_FACTORY_H_

#include "hybrid/node_executor/rts/rts_node_task.h"

namespace ge {
namespace hybrid {
using RtsNodeTaskPtr = std::shared_ptr<RtsNodeTask>;
using RtsTaskCreatorFun = std::function<RtsNodeTaskPtr()>;

class RtsTaskFactory {
 public:
  static RtsTaskFactory &GetInstance() {
    static RtsTaskFactory instance;
    return instance;
  }

  RtsNodeTaskPtr Create(const std::string &task_type) const;

  class RtsTaskRegistrar {
   public:
    RtsTaskRegistrar(const std::string &task_type, const RtsTaskCreatorFun &creator) {
      RtsTaskFactory::GetInstance().RegisterCreator(task_type, creator);
    }
    ~RtsTaskRegistrar() = default;
  };

 private:
  RtsTaskFactory() = default;
  ~RtsTaskFactory() = default;

  /**
   * Register build of executor
   * @param executor_type   type of executor
   * @param builder         build function
   */
  void RegisterCreator(const std::string &task_type, const RtsTaskCreatorFun &creator);

  std::map<std::string, RtsTaskCreatorFun> creators_;
};
}  // namespace hybrid
}  // namespace ge

#define REGISTER_RTS_TASK_CREATOR(task_type, task_clazz) \
    REGISTER_RTS_TASK_CREATOR_UNIQ_HELPER(__COUNTER__, task_type, task_clazz)

#define REGISTER_RTS_TASK_CREATOR_UNIQ_HELPER(ctr, type, clazz) \
  RtsTaskFactory::RtsTaskRegistrar g_##type##_Creator##ctr(type, []()->RtsNodeTaskPtr { return MakeShared<clazz>(); })

#endif // GE_HYBRID_NODE_EXECUTOR_RTS_TASK_FACTORY_H_
