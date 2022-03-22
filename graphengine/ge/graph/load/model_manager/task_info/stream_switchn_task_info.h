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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_STREAM_SWITCHN_TASK_INFO_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_STREAM_SWITCHN_TASK_INFO_H_

#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/op_desc.h"

namespace ge {
class StreamSwitchNTaskInfo : public TaskInfo {
 public:
  StreamSwitchNTaskInfo()
      : input_ptr_(nullptr),
        input_size_(0),
        value_ptr_(nullptr),
        true_stream_ptr_(nullptr),
        element_size_(0),
        data_type_(RT_SWITCH_INT64),
        args_offset_(0) {}

  ~StreamSwitchNTaskInfo() override {}

  Status Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) override;

  Status Distribute() override;

  Status CalculateArgs(const domi::TaskDef &task_def, DavinciModel *davinci_model) override;

 private:
  Status GetTrueStreamPtr(const OpDescPtr &op_desc, DavinciModel *davinci_model);
  Status InputPtrUpdate(const OpDescPtr &op_desc, DavinciModel *davinci_model);
  void *input_ptr_;
  uint32_t input_size_;
  void *value_ptr_;
  rtStream_t *true_stream_ptr_;
  uint32_t element_size_;
  rtSwitchDataType_t data_type_;
  vector<rtStream_t> true_stream_list_;
  vector<int64_t> value_list_;
  int64_t args_offset_;
};
}
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_STREAM_SWITCHN_TASK_INFO_H_
