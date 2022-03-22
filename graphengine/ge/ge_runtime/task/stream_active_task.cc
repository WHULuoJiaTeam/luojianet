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

#include "ge_runtime/task/stream_active_task.h"
#include "ge_runtime/task/task_factory.h"

namespace ge {
namespace model_runner {
StreamActiveTask::StreamActiveTask(const ModelContext &model_context,
                                   const std::shared_ptr<StreamActiveTaskInfo> &task_info)
    : TaskRepeater<StreamActiveTaskInfo>(model_context, task_info),
      task_info_(task_info),
      stream_(nullptr),
      active_stream_(nullptr) {
  if (task_info_ == nullptr) {
    GELOGW("task_info_ is null!");
    return;
  }
  auto stream_list = model_context.stream_list();
  uint32_t stream_id = task_info->stream_id();
  uint32_t active_stream_id = task_info->active_stream_id();
  GELOGI("Stream list size:%zu, stream id:%u, active stream id:%u", stream_list.size(), stream_id, active_stream_id);
  if (stream_id >= stream_list.size() || active_stream_id >= stream_list.size()) {
    GELOGW("Stream id invalid");
    return;
  }
  stream_ = stream_list[stream_id];
  active_stream_ = stream_list[active_stream_id];
}

StreamActiveTask::~StreamActiveTask() {}

bool StreamActiveTask::Distribute() {
  GELOGI("Distribute start");
  GELOGI("Stream %u active %u.", task_info_->stream_id(), task_info_->active_stream_id());
  rtError_t rt_ret = rtStreamActive(active_stream_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return false;
  }
  GELOGI("DistributeTask end");
  return true;
}

REGISTER_TASK(TaskInfoType::STREAM_ACTIVE, StreamActiveTask, StreamActiveTaskInfo);
}  // namespace model_runner
}  // namespace ge
