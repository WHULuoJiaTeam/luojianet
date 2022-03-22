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

#include "ge_runtime/task/label_goto_task.h"
#include "ge_runtime/task/task_factory.h"

namespace ge {
namespace model_runner {
LabelGotoTask::LabelGotoTask(const ModelContext &model_context, const std::shared_ptr<LabelGotoTaskInfo> &task_info)
    : TaskRepeater<LabelGotoTaskInfo>(model_context, task_info),
      task_info_(task_info),
      stream_(nullptr),
      index_value_(nullptr) {
  if (task_info_ == nullptr) {
    GELOGW("task_info_ is null!");
    return;
  }
  auto stream_list = model_context.stream_list();
  auto label_list = model_context.label_list();
  rt_model_handle_ = model_context.rt_model_handle();
  uint32_t stream_id = task_info->stream_id();
  label_id_ = task_info->label_id();
  GELOGI("Stream list size:%zu, stream id:%u.", stream_list.size(), stream_id);
  GELOGI("Label list size:%zu, label id:%u.", label_list.size(), label_id_);
  if (stream_id >= stream_list.size() || label_id_ >= label_list.size()) {
    GELOGW("Stream/Label id invalid.");
    return;
  }
  stream_ = stream_list[stream_id];
  label_manager_ = LabelManager::GetInstance();
  if (label_manager_ == nullptr) {
    GELOGW("Get label manager instance failed.");
    return;
  }
  label_info_ = label_manager_->GetLabelInfo(rt_model_handle_, {label_id_}, label_list);
}

LabelGotoTask::~LabelGotoTask() {
  if (index_value_ != nullptr) {
    rtError_t rt_ret = rtFree(index_value_);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "rtFree index_value_ failed! ret: 0x%X.", rt_ret);
    }
    index_value_ = nullptr;
  }
}

bool LabelGotoTask::Distribute() {
  GELOGI("LabelGotoTask Distribute start.");
  if (!CheckParamValid()) {
    return false;
  }

  const std::vector<void *> label_list = { label_ };
  rtError_t rt_ret = rtMalloc(&index_value_, sizeof(uint64_t), RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: %#x", rt_ret);
    return false;
  }

  uint64_t branch_index = 0;
  rt_ret = rtMemcpy(index_value_, sizeof(uint64_t), &branch_index, sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: %#x", rt_ret);
    return false;
  }

  uint32_t label_info_size = sizeof(rtLabelDevInfo) * label_list.size();
  rt_ret = rtMalloc(&label_info_, label_info_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: %#x", rt_ret);
    return false;
  }

  rt_ret = rtLabelListCpy(reinterpret_cast<void**>(label_list.data()), label_list.size(), label_info_, label_info_size);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: %#x", rt_ret);
    return false;
  }

  rt_ret = rtLabelSwitchByIndex(index_value_, label_list.size(), label_info_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: %#x", rt_ret);
    return false;
  }

  GELOGI("DistributeTask end.");
  return true;
}

bool LabelGotoTask::CheckParamValid() {
  if (stream_ == nullptr) {
    GELOGE(PARAM_INVALID, "stream is null!");
    return false;
  }

  if (label_info_ == nullptr) {
    GELOGE(PARAM_INVALID, "label info is null!");
    return false;
  }

  if (index_value_ == nullptr) {
    rtError_t rt_ret = rtMalloc(&index_value_, sizeof(uint64_t), RT_MEMORY_HBM);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return false;
    }

    uint64_t index = 0;
    rt_ret = rtMemcpy(index_value_, sizeof(uint64_t), &index, sizeof(index), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return false;
    }
  }

  void *label_info = label_info_->GetLabelInfo();
  rtError_t rt_ret = rtLabelSwitchByIndex(index_value_, 1, label_info, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return false;
  }

  GELOGI("DistributeTask end.");
  return true;
}

REGISTER_TASK(TaskInfoType::LABEL_GOTO, LabelGotoTask, LabelGotoTaskInfo);
}  // namespace model_runner
}  // namespace ge
