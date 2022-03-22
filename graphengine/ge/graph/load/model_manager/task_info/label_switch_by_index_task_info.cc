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

#include "graph/load/model_manager/task_info/label_switch_by_index_task_info.h"

#include "graph/load/model_manager/davinci_model.h"

namespace ge {
constexpr uint8_t kLabelSwitchIndexNum = 1;

LabelSwitchByIndexTaskInfo::~LabelSwitchByIndexTaskInfo() {
  GE_FREE_RT_LOG(args_);
  index_value_ = nullptr;
}

Status LabelSwitchByIndexTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("LabelSwitchByIndexTaskInfo Init Start.");
  GE_CHECK_NOTNULL(davinci_model);

  Status ret = SetStream(task_def.stream_id(), davinci_model->GetStreamList());
  if (ret != SUCCESS) {
    return FAILED;
  }

  // Get LabelSwitchByIndex task def
  const domi::LabelSwitchByIndexDef &label_switch = task_def.label_switch_by_index();
  OpDescPtr op_desc = davinci_model->GetOpByIndex(label_switch.op_index());
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "Can't get op_desc from davinci_model by index:%u", label_switch.op_index());
    GELOGE(INTERNAL_ERROR, "[Get][Op] Task op index:%u out of range!", label_switch.op_index());
    return INTERNAL_ERROR;
  }

  branch_max_ = label_switch.label_max();

  auto input_data_addr = ModelUtils::GetInputDataAddrs(davinci_model->GetRuntimeParam(), op_desc);
  if (input_data_addr.size() != kLabelSwitchIndexNum) {
    REPORT_INNER_ERROR("E19999", "input_data_addr size:%zu != kLabelSwitchIndexNum:%u, op:%s(%s), "
                       "check invalid", input_data_addr.size(), kLabelSwitchIndexNum,
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] %s invalid addr size:%zu, num:%u!",
           op_desc->GetName().c_str(), input_data_addr.size(), kLabelSwitchIndexNum);
    return INTERNAL_ERROR;
  }

  if (davinci_model->IsKnownNode()) {
    index_value_ = davinci_model->GetCurrentFixedAddr(fixed_addr_offset_);
  } else {
    index_value_ = input_data_addr[0];
  }

  davinci_model->DisableZeroCopy(index_value_);

  vector<uint32_t> label_idx_list;
  if (!AttrUtils::GetListInt(op_desc, ATTR_NAME_LABEL_SWITCH_LIST, label_idx_list)) {
    REPORT_INNER_ERROR("E19999", "Get Attr:%s in op:%s(%s) fail",
                       ATTR_NAME_LABEL_SWITCH_LIST.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][Attr] %s in op:%s(%s) failed.", ATTR_NAME_LABEL_SWITCH_LIST.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return INTERNAL_ERROR;
  }

  if (label_idx_list.empty() || label_idx_list.size() != branch_max_) {
    REPORT_INNER_ERROR("E19999", "label_idx_list in op:%s(%s) is empty, or size:%zu != branch_max_:%u"
                       "check invalid", op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                       label_idx_list.size(), branch_max_);
    GELOGE(INTERNAL_ERROR, "[Check][Param] %s label index size:%zu, task branch max:%u.",
           op_desc->GetName().c_str(), label_idx_list.size(), branch_max_);
    return INTERNAL_ERROR;
  }

  vector<rtLabel_t> label_used(branch_max_, nullptr);
  const vector<rtLabel_t> &label_list = davinci_model->GetLabelList();
  for (size_t idx = 0; idx < label_idx_list.size(); ++idx) {
    uint32_t label_id = label_idx_list[idx];
    if (label_id >= label_list.size()) {
      REPORT_INNER_ERROR("E19999", "label_id:%u in op:%s(%s) >= label_list.size():%zu in model"
                         "check invalid", label_id,
                         op_desc->GetName().c_str(), op_desc->GetType().c_str(), label_list.size());
      GELOGE(INTERNAL_ERROR, "[Check][Param] %s index:%zu, label index:%u, model label size:%zu.",
             op_desc->GetName().c_str(), idx, label_id, label_list.size());
      return INTERNAL_ERROR;
    }
    GE_CHECK_NOTNULL(label_list[label_id]);
    label_used[idx] = label_list[label_id];
  }

  rtMemType_t memory_type = op_desc->HasAttr(ATTR_NAME_MEMORY_TYPE_RANGE) ? RT_MEMORY_TS_4G : RT_MEMORY_HBM;
  GELOGI("memory_type: %u", memory_type);
  args_size_ = branch_max_ * sizeof(rtLabelDevInfo);
  rtError_t rt_ret = rtMalloc(&args_, args_size_, memory_type);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed for op:%s(%s), size:%u, ret:0x%X",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), args_size_, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMalloc] failed for op:%s(%s), size:%u, ret:0x%X",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), args_size_, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  rt_ret = rtLabelListCpy(label_used.data(), label_used.size(), args_, args_size_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtLabelListCpy failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtLabelListCpy] failed, ret:0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  GELOGI("LabelSwitchByIndexTaskInfo Init success, branch max: %u.", branch_max_);
  return SUCCESS;
}

Status LabelSwitchByIndexTaskInfo::Distribute() {
  GELOGI("LabelSwitchByIndexTaskInfo Distribute Start, branch max: %u", branch_max_);
  GE_CHECK_NOTNULL(args_);
  GE_CHECK_NOTNULL(index_value_);
  if (branch_max_ == 0 || args_size_ == 0) {
    REPORT_INNER_ERROR("E19999", "branch_max_:%u or args_size_:%u is 0, check invalid", branch_max_, args_size_);
    GELOGE(PARAM_INVALID, "[Check][Param] branch max:%u, args size:%u invalid.", branch_max_, args_size_);
    return PARAM_INVALID;
  }

  rtError_t rt_ret = rtLabelSwitchByIndex(index_value_, branch_max_, args_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtLabelSwitchByIndex failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtLabelSwitchByIndex] failed, ret:0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  GELOGI("LabelSwitchByIndexTaskInfo Distribute Success.");
  return SUCCESS;
}

Status LabelSwitchByIndexTaskInfo::CalculateArgs(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GE_CHECK_NOTNULL(davinci_model);
  auto label_switch = task_def.label_switch_by_index();
  uint32_t op_index = label_switch.op_index();
  GELOGI("Begin to calculate args, op_index is: %u", op_index);
  auto op_desc = davinci_model->GetOpByIndex(op_index);
  GE_CHECK_NOTNULL(op_desc);
  GELOGI("Calc opType[%s] args size. Node name is [%s]", op_desc->GetType().c_str(), op_desc->GetName().c_str());
  if (op_desc->GetInputsSize() != kLabelSwitchIndexNum) {
    REPORT_INNER_ERROR("E19999", "input size:%zu in op:%s(%s) != kLabelSwitchIndexNum"
                       "check invalid", op_desc->GetInputsSize(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Check][Param] Label switch op:%s(%s) only have one data input. Now input size is %zu",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), op_desc->GetInputsSize());
    return FAILED;
  }
  string input_tensor_name = op_desc->GetName();
  fixed_addr_offset_ = davinci_model->GetFixedAddrsSize(input_tensor_name);
  auto tensor_desc = op_desc->GetInputDesc(0);
  int64_t tensor_size = 0;
  GE_CHK_STATUS(TensorUtils::GetSize(tensor_desc, tensor_size));
  davinci_model->SetTotalFixedAddrsSize(input_tensor_name, tensor_size);
  GELOGI("Calculate stream switchn task args , tensor_size %ld, fixed_addr_offset %ld", tensor_size,
         fixed_addr_offset_);
  return SUCCESS;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_STREAM_LABEL_SWITCH_BY_INDEX, LabelSwitchByIndexTaskInfo);
}  // namespace ge
