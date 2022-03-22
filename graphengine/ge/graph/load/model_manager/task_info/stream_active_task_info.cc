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

#include "graph/load/model_manager/task_info/stream_active_task_info.h"

#include <vector>

#include "framework/common/debug/ge_log.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
Status StreamActiveTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("StreamActiveTaskInfo Init Start.");
  if (davinci_model == nullptr) {
    REPORT_INNER_ERROR("E19999", "Check param davinci_model nullptr");
    GELOGE(PARAM_INVALID, "[Check][Param] davinci_model is null!");
    return PARAM_INVALID;
  }

  Status ret = SetStream(task_def.stream_id(), davinci_model->GetStreamList());
  if (ret != SUCCESS) {
    return ret;
  }

  auto stream_active_def = task_def.stream_active();
  uint32_t op_index = stream_active_def.op_index();

  uint32_t internal_index = davinci_model->GetFlowctrlIndex(op_index);

  // get StreamActive op
  OpDescPtr op_desc = davinci_model->GetOpByIndex(op_index);
  GE_CHECK_NOTNULL(op_desc);
  std::vector<uint32_t> active_stream_index_list;
  if (!AttrUtils::GetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_index_list)) {
    REPORT_INNER_ERROR("E19999", "Get Attr:%s in op:%s(%s) fail", ATTR_NAME_ACTIVE_STREAM_LIST.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][Attr] %s in op:%s(%s) fail", ATTR_NAME_ACTIVE_STREAM_LIST.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return INTERNAL_ERROR;
  }

  if (internal_index >= active_stream_index_list.size()) {
    REPORT_INNER_ERROR("E19999", "flowctrl index:%u >= active_stream_list size:%zu in op:%s(%s), "
                       "check invalid", internal_index, active_stream_index_list.size(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] stream id index invalid. index:%u, list size:%zu, op:%s(%s).",
           internal_index, active_stream_index_list.size(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return INTERNAL_ERROR;
  }

  if (active_stream_index_list[internal_index] >= davinci_model->GetStreamList().size()) {
    REPORT_INNER_ERROR("E19999", "active_stream_index:%u in op:%s(%s) >= stream size:%zu in model, "
                       "check invalid", active_stream_index_list[internal_index],
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), davinci_model->GetStreamList().size());
    GELOGE(INTERNAL_ERROR, "[Check][Param] active_stream_index:%u in op:%s(%s) >= stream size:%zu in model",
           active_stream_index_list[internal_index], op_desc->GetName().c_str(), op_desc->GetType().c_str(),
           davinci_model->GetStreamList().size());
    return INTERNAL_ERROR;
  }

  active_stream_ = davinci_model->GetStreamList()[active_stream_index_list[internal_index]];
  active_stream_id_ = stream_active_def.active_stream_id();
  GELOGI("InitStreamActiveTaskInfo Init Success, index:%u, activeStream:%p, activeStreamID:%u.",
         internal_index, active_stream_, active_stream_id_);

  return SUCCESS;
}

Status StreamActiveTaskInfo::Distribute() {
  GELOGI("StreamActiveTaskInfo Distribute Start.");
  rtError_t rt_ret = rtStreamActive(active_stream_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtStreamActive failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtStreamActive] failed, ret:0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  GELOGI("StreamActiveTaskInfo Distribute Success. activeStreamID:%p.", active_stream_);
  return SUCCESS;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_STREAM_ACTIVE, StreamActiveTaskInfo);
}  // namespace ge
