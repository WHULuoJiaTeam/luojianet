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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_END_GRAPH_TASK_INFO_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_END_GRAPH_TASK_INFO_H_

#include "graph/load/model_manager/task_info/task_info.h"

namespace ge {
class EndGraphTaskInfo : public TaskInfo {
 public:
  EndGraphTaskInfo() {}

  ~EndGraphTaskInfo() override { model_ = nullptr; }

  Status Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) override;

  Status Distribute() override;

  uint32_t GetTaskID() override { return task_id_; }

  uint32_t GetStreamId() override { return stream_id_; }

 private:
  rtModel_t model_{nullptr};
  DavinciModel *davinci_model_{nullptr};
  uint32_t task_id_{0};
  uint32_t stream_id_{0};
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_END_GRAPH_TASK_INFO_H_
