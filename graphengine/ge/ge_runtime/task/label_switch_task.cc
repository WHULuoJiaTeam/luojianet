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

#include "ge_runtime/task/label_switch_task.h"
#include "ge_runtime/task/task_factory.h"

namespace ge {
namespace model_runner {
LabelSwitchTask::LabelSwitchTask(const ModelContext &model_context,
                                 const std::shared_ptr<LabelSwitchTaskInfo> &task_info)
    : TaskRepeater<LabelSwitchTaskInfo>(model_context, task_info),
      task_info_(task_info),
      stream_(nullptr),
      label_info_(nullptr) {
  if (task_info_ == nullptr) {
    GELOGW("task_info_ is null!");
    return;
  }

  rt_model_handle_ = model_context.rt_model_handle();
  auto all_label_resource = model_context.label_list();
  auto stream_list = model_context.stream_list();
  uint32_t stream_id = task_info->stream_id();
  GELOGI("Stream list size:%zu, stream id:%u.", stream_list.size(), stream_id);
  if (stream_id >= stream_list.size()) {
    GELOGW("Stream id invalid.");
    return;
  }
  stream_ = stream_list[stream_id];
  label_manager_ = LabelManager::GetInstance();
  if (label_manager_ == nullptr) {
    GELOGW("Get label manager instance failed.");
    return;
  }
  label_info_ = label_manager_->GetLabelInfo(rt_model_handle_, task_info_->label_list(), all_label_resource);
}

LabelSwitchTask::~LabelSwitchTask() {}

bool LabelSwitchTask::Distribute() {
  GELOGI("LabelSwitchTask Distribute start.");
  if (!CheckParamValid()) {
    return false;
  }

  const std::vector<uint32_t> &label_index_list = task_info_->label_list();
  std::vector<void *> label_list(task_info_->label_size(), nullptr);

  for (size_t i = 0; i < task_info_->label_size(); ++i) {
    uint32_t label_index = label_index_list[i];
    if (label_index >= all_label_resource_.size()) {
      GELOGE(PARAM_INVALID, "label %zu index is %u, but there are %zu labels in total.", i, label_index,
             all_label_resource_.size());
      return false;
    }
    label_list[i] = all_label_resource_[label_index];
    GELOGI("Case %zu: label id %zu.", i, (size_t)label_index);
  }

  uint32_t label_info_size = sizeof(rtLabelDevInfo) * task_info_->label_size();
  rtError_t rt_ret = rtMalloc(&label_info_, label_info_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return false;
  }

  rt_ret = rtLabelListCpy(label_list.data(), label_list.size(), label_info_, label_info_size);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return false;
  }

  rt_ret = rtLabelSwitchByIndex(task_info_->cond(), label_list.size(), label_info_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return false;
  }

  GELOGI("DistributeTask end.");
  return true;
}

bool LabelSwitchTask::CheckParamValid() {
  if (stream_ == nullptr) {
    GELOGE(PARAM_INVALID, "stream is null!");
    return false;
  }

  if (task_info_->label_list().empty()) {
    GELOGE(PARAM_INVALID, "label_list is empty.");
    return false;
  }

  if (task_info_->label_size() != task_info_->label_list().size()) {
    GELOGE(PARAM_INVALID, "label_list size %zu but label_size is %u.", task_info_->label_list().size(),
           task_info_->label_size());
    return false;
  }

  if (task_info_->label_size() >= UINT32_MAX / sizeof(rtLabelDevInfo)) {
    GELOGE(PARAM_INVALID, "label_size %u will overflow.", task_info_->label_size());
    return false;
  }

  if (label_info_ == nullptr) {
    GELOGE(PARAM_INVALID, "CopyLabelList failed, label info is null.");
    return false;
  }

  return true;
}

REGISTER_TASK(TaskInfoType::LABEL_SWITCH, LabelSwitchTask, LabelSwitchTaskInfo);
}  // namespace model_runner
}  // namespace ge
