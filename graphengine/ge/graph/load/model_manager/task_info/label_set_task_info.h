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

#ifndef GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_LABEL_SET_TASK_INFO_H_
#define GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_LABEL_SET_TASK_INFO_H_

#include "graph/load/model_manager/task_info/task_info.h"

namespace ge {
class LabelSetTaskInfo : public TaskInfo {
 public:
  LabelSetTaskInfo() : label_(nullptr) {}

  ~LabelSetTaskInfo() override { label_ = nullptr; }

  Status Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) override;

  Status Distribute() override;

 private:
  void *label_;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_LABEL_SET_TASK_INFO_H_
