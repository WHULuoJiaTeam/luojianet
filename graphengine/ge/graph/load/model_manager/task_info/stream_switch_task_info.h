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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_STREAM_SWITCH_TASK_INFO_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_STREAM_SWITCH_TASK_INFO_H_
#include "graph/load/model_manager/task_info/task_info.h"

namespace ge {
class StreamSwitchTaskInfo : public TaskInfo {
 public:
  StreamSwitchTaskInfo()
      : input_ptr_(nullptr),
        cond_(RT_EQUAL),
        value_ptr_(nullptr),
        true_stream_(nullptr),
        true_stream_id_(0),
        data_type_(RT_SWITCH_INT32) {}

  ~StreamSwitchTaskInfo() override {
    input_ptr_ = nullptr;
    value_ptr_ = nullptr;
    true_stream_ = nullptr;
  }

  Status Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) override;

  Status Distribute() override;

  Status CalculateArgs(const domi::TaskDef &task_def, DavinciModel *davinci_model) override;
 private:
  void SetInputAndValuePtr(DavinciModel *davinci_model, const std::vector<void *> &input_data_addrs);
  void *input_ptr_;
  rtCondition_t cond_;
  void *value_ptr_;
  rtStream_t true_stream_;
  uint32_t true_stream_id_;
  rtSwitchDataType_t data_type_;
  static const uint32_t kInputNum = 2;
  std::vector<int64_t> fixed_addr_offset_;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_STREAM_SWITCH_TASK_INFO_H_
