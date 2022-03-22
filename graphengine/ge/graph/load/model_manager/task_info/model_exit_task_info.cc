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

#include "graph/load/model_manager/task_info/model_exit_task_info.h"

#include "common/properties_manager.h"
#include "framework/common/debug/ge_log.h"
#include "graph/load/model_manager/davinci_model.h"

namespace ge {
Status ModelExitTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("InitModelExitTaskInfo Init Start.");
  if (davinci_model == nullptr) {
    REPORT_INNER_ERROR("E19999", "Check param davinci_model nullptr");
    GELOGE(PARAM_INVALID, "[Check][Param] davinci_model is null!");
    return PARAM_INVALID;
  }

  Status ret = SetStream(task_def.stream_id(), davinci_model->GetStreamList());
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][Stream] fail, stream_id:%u", task_def.stream_id());
    return ret;
  }

  model_ = davinci_model->GetRtModelHandle();
  GELOGI("InitModelExitTaskInfo Init Success, model:%p, stream:%p", model_, stream_);
  return SUCCESS;
}

Status ModelExitTaskInfo::Distribute() {
  GELOGI("ModelExitTaskInfo Distribute Start.");
  rtError_t rt_ret = rtModelExit(model_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtModelExit failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtModelExit] failed, ret:0x%x", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  GELOGI("ModelExitTaskInfo Distribute Success.");
  return SUCCESS;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_MODEL_EXIT, ModelExitTaskInfo);
}  // namespace ge
