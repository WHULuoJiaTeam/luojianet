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

#include "ge_runtime/task/label_set_task.h"
#include "ge_runtime/task/task_factory.h"

namespace ge {
namespace model_runner {
LabelSetTask::LabelSetTask(const ModelContext &model_context, const std::shared_ptr<LabelSetTaskInfo> &task_info)
    : TaskRepeater<LabelSetTaskInfo>(model_context, task_info),
      task_info_(task_info),
      stream_(nullptr),
      label_(nullptr) {
  if (task_info_ == nullptr) {
    GELOGW("task_info_ is null!");
    return;
  }
  auto stream_list = model_context.stream_list();
  auto label_list = model_context.label_list();
  uint32_t stream_id = task_info->stream_id();
  uint32_t label_id = task_info->label_id();
  GELOGI("Stream list size:%zu, stream id:%u.", stream_list.size(), stream_id);
  GELOGI("Label list size:%zu, label id:%u.", label_list.size(), label_id);
  if (stream_id >= stream_list.size() || label_id >= label_list.size()) {
    GELOGW("Stream/Label id invalid.");
    return;
  }
  stream_ = stream_list[stream_id];
  label_ = label_list[label_id];
}

LabelSetTask::~LabelSetTask() {}

bool LabelSetTask::Distribute() {
  GELOGI("LabelSetTask Distribute start.");
  if (stream_ == nullptr) {
    GELOGE(PARAM_INVALID, "stream is null!");
    return false;
  }
  if (label_ == nullptr) {
    GELOGE(PARAM_INVALID, "label is null!");
    return false;
  }
  rtError_t rt_ret = rtLabelSet(label_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return false;
  }

  GELOGI("DistributeTask end.");
  return true;
}

REGISTER_TASK(TaskInfoType::LABEL_SET, LabelSetTask, LabelSetTaskInfo);

}  // namespace model_runner
}  // namespace ge
