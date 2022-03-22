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

#include "graph/load/model_manager/task_info/label_goto_ex_task_info.h"

#include "graph/load/model_manager/davinci_model.h"

namespace ge {
constexpr uint8_t kGotoBranchMax = 1;

LabelGotoExTaskInfo::~LabelGotoExTaskInfo() {
  args_ = nullptr;
  GE_FREE_RT_LOG(index_value_);
}

Status LabelGotoExTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("LabelGotoExTaskInfo Init Start.");
  GE_CHECK_NOTNULL(davinci_model);

  if (SetStream(task_def.stream_id(), davinci_model->GetStreamList()) != SUCCESS) {
    return FAILED;
  }

  // Get LabelGotoEx task def
  const domi::LabelGotoExDef &label_goto = task_def.label_goto_ex();
  OpDescPtr op_desc = davinci_model->GetOpByIndex(label_goto.op_index());
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "Can't get op_desc from davinci_model by index:%u",
                       label_goto.op_index());
    GELOGE(INTERNAL_ERROR, "[Get][Op] Task op index:%u out of range!", label_goto.op_index());
    return INTERNAL_ERROR;
  }

  uint32_t label_index = 0;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, label_index)) {
    REPORT_INNER_ERROR("E19999", "Get Attr:%s in op:%s(%s) fail",
                       ATTR_NAME_LABEL_SWITCH_INDEX.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][Attr] %s in op:%s(%s) fail.",
           ATTR_NAME_LABEL_SWITCH_INDEX.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return INTERNAL_ERROR;
  }

  rtMemType_t memory_type = op_desc->HasAttr(ATTR_NAME_MEMORY_TYPE_RANGE) ? RT_MEMORY_TS_4G : RT_MEMORY_HBM;
  GELOGI("memory_type: %u", memory_type);

  GE_CHK_STATUS_RET_NOLOG(davinci_model->GetLabelGotoAddr(label_index, memory_type, args_, args_size_));

  rtError_t rt_ret = rtMalloc(&index_value_, sizeof(uint64_t), memory_type);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed for op:%s(%s), size:%lu, ret:0x%X",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), sizeof(uint64_t), rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMalloc] failed for op:%s(%s), size:%lu, ret:0x%X",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), sizeof(uint64_t), rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  uint64_t branch_index = 0;
  rt_ret = rtMemcpy(index_value_, sizeof(uint64_t), &branch_index, sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed for op:%s(%s), size:%lu, ret:0x%X",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), sizeof(uint64_t), rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed for op:%s(%s), size:%lu, ret:0x%X",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), sizeof(uint64_t), rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  GELOGI("LabelGotoExTaskInfo Init Success, label id:%u", label_index);
  return SUCCESS;
}

Status LabelGotoExTaskInfo::Distribute() {
  GELOGI("LabelGotoExTaskInfo Distribute Start.");
  GE_CHECK_NOTNULL(args_);
  GE_CHECK_NOTNULL(index_value_);
  if (args_size_ == 0) {
    REPORT_INNER_ERROR("E19999", "Param args_size_ is 0, check fail");
    GELOGE(PARAM_INVALID, "[Check][Param] branch max:%u, args size:%u invalid.", kGotoBranchMax, args_size_);
    return PARAM_INVALID;
  }

  rtError_t rt_ret = rtLabelSwitchByIndex(index_value_, kGotoBranchMax, args_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtLabelSwitchByIndex failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtLabelSwitchByIndex] failed, ret:0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  GELOGI("LabelGotoExTaskInfo Distribute Success.");
  return SUCCESS;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_STREAM_LABEL_GOTO, LabelGotoExTaskInfo);
}  // namespace ge
