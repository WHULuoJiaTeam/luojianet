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

#include "graph/load/model_manager/task_info/end_graph_task_info.h"

#include "common/properties_manager.h"
#include "framework/common/debug/ge_log.h"
#include "graph/load/model_manager/davinci_model.h"

namespace {
const uint32_t kDumpFlag = 2;
}  // namespace
namespace ge {
Status EndGraphTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("InitEndGraphTaskInfo Init Start.");
  if (davinci_model == nullptr) {
    REPORT_INNER_ERROR("E19999", "Check param davinci_model nullptr");
    GELOGE(PARAM_INVALID, "[Check][Param] davinci_model is null!");
    return PARAM_INVALID;
  }
  davinci_model_ = davinci_model;
  Status ret = SetStream(task_def.stream_id(), davinci_model->GetStreamList());
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][Stream] fail, stream_id:%u", task_def.stream_id());
    return ret;
  }

  model_ = davinci_model->GetRtModelHandle();
  GELOGI("InitEndGraphTaskInfo Init Success, model:%p, stream:%p", model_, stream_);
  return SUCCESS;
}

Status EndGraphTaskInfo::Distribute() {
  GELOGI("EndGraphTaskInfo Distribute Start.");
  GE_CHECK_NOTNULL(davinci_model_);
  if (davinci_model_->ModelNeedDump()) {
    GELOGI("Start to call rtEndGraphEx");
    rtError_t rt_ret = rtEndGraphEx(model_, stream_, kDumpFlag);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtEndGraphEx failed, ret:0x%X", rt_ret);
      GELOGE(RT_FAILED, "[Call][RtEndGraphEx] failed, ret:0x%x", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
  } else {
    GELOGI("Start to call rtEndGraph");
    rtError_t rt_ret = rtEndGraph(model_, stream_);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtEndGraph failed, ret:0x%X", rt_ret);
      GELOGE(RT_FAILED, "[Call][RtEndGraph] failed, ret:0x%x", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
  }

  uint32_t task_id = 0;
  uint32_t stream_id = 0;
  rtError_t rt_ret = rtModelGetTaskId(davinci_model_->GetRtModelHandle(), &task_id, &stream_id);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtModelGetTaskId failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtModelGetTaskId] failed, ret:0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  task_id_ = task_id;
  stream_id_ = stream_id;
  davinci_model_->SetEndGraphId(task_id, stream_id);

  GELOGI("EndGraphTaskInfo Distribute Success, task id is %u, stream id is %u", task_id, stream_id);
  return SUCCESS;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_MODEL_END_GRAPH, EndGraphTaskInfo);
}  // namespace ge
