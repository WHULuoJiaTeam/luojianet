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

#include "ge_runtime/task/stream_switch_task.h"
#include "ge_runtime/task/task_factory.h"

namespace ge {
namespace model_runner {
StreamSwitchTask::StreamSwitchTask(const ModelContext &model_context,
                                   const std::shared_ptr<StreamSwitchTaskInfo> &task_info)
    : TaskRepeater<StreamSwitchTaskInfo>(model_context, task_info),
      task_info_(task_info),
      stream_(nullptr),
      stream_list_() {
  if (task_info_ == nullptr) {
    GELOGW("task_info_ is null!");
    return;
  }

  stream_list_ = model_context.stream_list();
  if (stream_list_.size() == 1) {
    stream_ = stream_list_[0];
  } else if (stream_list_.size() > task_info->stream_id()) {
    stream_ = stream_list_[task_info->stream_id()];
  } else {
    GELOGW("Index: %u >= stream_list.size(): %zu.", task_info->stream_id(), stream_list_.size());
  }
}

StreamSwitchTask::~StreamSwitchTask() {}

bool StreamSwitchTask::Distribute() {
  GELOGI("Init StreamSwitchTask start.");
  GELOGI("Stream %u active %ld.", task_info_->stream_id(), task_info_->true_stream_id());

  if (stream_ == nullptr) {
    GELOGE(PARAM_INVALID, "stream_ is null!");
    return false;
  }

  if (static_cast<uint64_t>(task_info_->true_stream_id()) >= stream_list_.size()) {
    GELOGE(PARAM_INVALID, "true_stream_id %ld must be less than stream_list_ size %zu!", task_info_->true_stream_id(),
           stream_list_.size());
    return false;
  }

  void *input = reinterpret_cast<void *>(task_info_->input_addr());
  rtCondition_t cond = static_cast<rtCondition_t>(task_info_->cond());
  void *value = reinterpret_cast<void *>(task_info_->value_addr());
  rtStream_t true_stream = stream_list_[task_info_->true_stream_id()];
  rtSwitchDataType_t data_type = static_cast<rtSwitchDataType_t>(task_info_->data_type());

  GELOGI("InitStreamSwitchTask, cond:%d, trueStream:%p, trueStreamID:%ld, datatype:%ld.", cond, true_stream,
         task_info_->true_stream_id(), task_info_->data_type());

  GELOGI("StreamSwitchTask Distribute Start.");
  rtError_t rt_ret = rtStreamSwitchEx(input, cond, value, true_stream, stream_, data_type);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return false;
  }

  GELOGI("Distribute StreamSwitch, cond:%d, trueStream:%p, datatype:%ld.", cond, true_stream, task_info_->data_type());
  return true;
}

REGISTER_TASK(TaskInfoType::STREAM_SWITCH, StreamSwitchTask, StreamSwitchTaskInfo);
}  // namespace model_runner
}  // namespace ge
