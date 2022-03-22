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
#include "graph/load/model_manager/task_info/stream_switchn_task_info.h"
#include <vector>
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_utils.h"

namespace {
const uint8_t kStreamSwitchnInputNum = 1;
}

namespace ge {
Status StreamSwitchNTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("StreamSwitchNTaskInfo Init Start.");
  GE_CHECK_NOTNULL(davinci_model);

  if (SetStream(task_def.stream_id(), davinci_model->GetStreamList()) != SUCCESS) {
    return FAILED;
  }

  auto stream_switchn_def = task_def.stream_switch_n();
  OpDescPtr op_desc = davinci_model->GetOpByIndex(stream_switchn_def.op_index());
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "Can't get op_desc from davinci_model by index:%u", stream_switchn_def.op_index());
    GELOGE(FAILED, "[Get][Op] failed, as Index is out of range, index:%u", stream_switchn_def.op_index());
    return FAILED;
  }

  // set size_
  input_size_ = stream_switchn_def.size();

  // set value_ptr_
  auto value = stream_switchn_def.target_value();
  if (value.size() == 0) {
    REPORT_INNER_ERROR("E19999", "task_Def.stream_switch_n.target_value:%d in op:%s(%s) is 0,"
                       "check invalid", value.size(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Check][Param] The number of gears in dynamic batch scenario can not be 0, op:%s.",
           op_desc->GetName().c_str());
    return FAILED;
  }
  for (int i = 0; i < value.size(); ++i) {
    GELOGD("InitStreamSwitchTaskInfo, valuePtr value[%d]: %ld.", i, value[i]);
    value_list_.emplace_back(value[i]);
  }
  value_ptr_ = &value_list_[0];

  // set element_size_
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_BATCH_NUM, element_size_)) {
    REPORT_INNER_ERROR("E19999", "Get Attr:%s in op:%s(%s) fail", ATTR_NAME_BATCH_NUM.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Get][Attr] %s in op:%s(%s) fail",
           ATTR_NAME_BATCH_NUM.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }

  if (GetTrueStreamPtr(op_desc, davinci_model) != SUCCESS) {
    GELOGE(FAILED, "[Get][TrueStreamPtr] of switchN op:%s failed.", op_desc->GetName().c_str());
    return FAILED;
  }

  // update StreamSwitchN's input_ptr_
  Status ret = InputPtrUpdate(op_desc, davinci_model);
  if (ret != SUCCESS) {
    return ret;
  }

  davinci_model->DisableZeroCopy(input_ptr_);
  GELOGI("StreamSwitchNTaskInfo Init Success, inputSize:%u, elementSize:%d, trueStreamID:%ld.", input_size_,
         element_size_, op_desc->GetStreamId());

  return SUCCESS;
}

Status StreamSwitchNTaskInfo::Distribute() {
  GELOGI("StreamSwitchNTaskInfo Distribute Start.");
  rtError_t rt_ret =
      rtStreamSwitchN(input_ptr_, input_size_, value_ptr_, true_stream_ptr_, element_size_, stream_, data_type_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtStreamSwitchN failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtStreamSwitchN] failed, ret:0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  GELOGI("StreamSwitchNTaskInfo Distribute Success. inputSize:%u, elementSize:%d, datatype:%d.", input_size_,
         element_size_, data_type_);
  return SUCCESS;
}

Status StreamSwitchNTaskInfo::GetTrueStreamPtr(const OpDescPtr &op_desc, DavinciModel *davinci_model) {
  vector<uint32_t> true_stream_id_list;
  if (!AttrUtils::GetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, true_stream_id_list)) {
    REPORT_INNER_ERROR("E19999", "Get Attr:%s in op:%s(%s) fail", ATTR_NAME_ACTIVE_STREAM_LIST.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Get][Attr] %s in op:%s(%s) fail",
           ATTR_NAME_ACTIVE_STREAM_LIST.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }

  if (true_stream_id_list.size() > davinci_model->GetStreamList().size()) {
    REPORT_INNER_ERROR("E19999", "active_stream_list.size:%zu in op:%s(%s) >= stream list size:%zu in model,"
                       "check invalid", true_stream_id_list.size(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), davinci_model->GetStreamList().size());
    GELOGE(FAILED,
           "[Check][Param] InitStreamSwitchNTaskInfo get true stream id list failed. true stream size:%zu, "
           "stream list size:%zu.", true_stream_id_list.size(), davinci_model->GetStreamList().size());
    return FAILED;
  }

  // set true_stream_ptr_
  for (size_t i = 0; i < true_stream_id_list.size(); ++i) {
    uint32_t true_stream_id = true_stream_id_list[i];
    if (true_stream_id >= davinci_model->GetStreamList().size()) {
      REPORT_INNER_ERROR("E19999", "active_stream_id:%u in op:%s(%s) >= stream list size:%zu in model,"
                         "check invalid", true_stream_id,
                         op_desc->GetName().c_str(), op_desc->GetType().c_str(), davinci_model->GetStreamList().size());
      GELOGE(FAILED, " [Check][Param] stream id:%u in op:%s invalid, stream list size:%zu.",
             true_stream_id, op_desc->GetName().c_str(), davinci_model->GetStreamList().size());
      return FAILED;
    }
    rtStream_t true_stream = davinci_model->GetStreamList()[true_stream_id];
    true_stream_list_.emplace_back(true_stream);
    GELOGD("InitStreamSwitchTaskInfo, trueStreamList index: %zu.", i);
  }

  if (true_stream_list_.empty()) {
    REPORT_INNER_ERROR("E19999", "active_stream_list.size():%zu in op:%s(%s) is empty, "
                       "check invalid", true_stream_id_list.size(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Check][Param] true stream list is null, op:%s.", op_desc->GetName().c_str());
    return FAILED;
  }
  true_stream_ptr_ = &true_stream_list_[0];
  return SUCCESS;
}

Status StreamSwitchNTaskInfo::CalculateArgs(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GE_CHECK_NOTNULL(davinci_model);
  auto stream_switchn_def = task_def.stream_switch_n();
  uint32_t op_index = stream_switchn_def.op_index();
  GELOGI("Begin to calculate args, op_index is: %u", op_index);
  auto op_desc = davinci_model->GetOpByIndex(op_index);
  GE_CHECK_NOTNULL(op_desc);
  GELOGI("Calc opType[%s] args size. Node name is [%s]", op_desc->GetType().c_str(), op_desc->GetName().c_str());
  if (op_desc->GetInputsSize() != kStreamSwitchnInputNum) {
    REPORT_INNER_ERROR("E19999", "input size:%zu in op:%s(%s) != kStreamSwitchnInputNum:%u, "
                       "check invalid", op_desc->GetInputsSize(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), kStreamSwitchnInputNum);
    GELOGE(FAILED, "[Check][Param] Stream switchn op:%s only have one data input. Now input size is %zu",
           op_desc->GetName().c_str(), op_desc->GetInputsSize());
    return FAILED;
  }
  string input_tensor_name = op_desc->GetInputNameByIndex(0);
  args_offset_ = davinci_model->GetFixedAddrsSize(input_tensor_name);
  auto tensor_desc = op_desc->GetInputDesc(0);
  int64_t tensor_size = 0;
  GE_CHK_STATUS(TensorUtils::GetSize(tensor_desc, tensor_size));
  davinci_model->SetTotalFixedAddrsSize(input_tensor_name, tensor_size);
  GELOGI("Calculate stream switchn task args, tensor_size %ld, args_offset %ld", tensor_size, args_offset_);
  return SUCCESS;
}

Status StreamSwitchNTaskInfo::InputPtrUpdate(const OpDescPtr &op_desc, DavinciModel *davinci_model) {
  // dst_ needs different address for different chips
  vector<int64_t> memory_type_list;
  (void)AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, memory_type_list);
  if (!memory_type_list.empty() && memory_type_list[0] == RT_MEMORY_TS_4G) {    // TS Feature, Just one.
    const vector<int64_t> input_offset = op_desc->GetInputOffset();
    const vector<int64_t> input_legnth = ModelUtils::GetInputSize(op_desc);
    if (input_offset.empty() || input_legnth.empty()) {
      REPORT_INNER_ERROR("E19999", "input_offset size:%zu or input_length.size:%zu in op:%s(%s) is empty,"
                         "check invalid", input_offset.size(), input_legnth.size(),
                         op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(FAILED, "[Check][Param] op:%s input offset size %zu, input legnth size:%zu",
             op_desc->GetName().c_str(), input_offset.size(), input_legnth.size());
      return FAILED;
    }
    const RuntimeParam &rts_param = davinci_model->GetRuntimeParam();
    input_ptr_ = rts_param.ts_mem_mall->Acquire(input_offset[0], input_legnth[0]);
  } else {
    if (davinci_model->IsKnownNode()) {
      input_ptr_ = davinci_model->GetCurrentFixedAddr(args_offset_);
    } else {
      auto input_data_addr = ModelUtils::GetInputDataAddrs(davinci_model->GetRuntimeParam(), op_desc);
      if (input_data_addr.empty()) {
        REPORT_INNER_ERROR("E19999", "input_data_addr size:%zu in op:%s(%s) is empty,"
                           "check invalid", input_data_addr.size(),
                           op_desc->GetName().c_str(), op_desc->GetType().c_str());
        GELOGE(FAILED, "[Check][Param] input data addr is empty in op:%s", op_desc->GetName().c_str());
        return FAILED;
      }
      input_ptr_ = input_data_addr[0];
    }
  }

  GELOGI("StreamSwitchN's input_ptr is %p", input_ptr_);
  return SUCCESS;
}
REGISTER_TASK_INFO(RT_MODEL_TASK_STREAM_SWITCH_N, StreamSwitchNTaskInfo);
}  // namespace ge
