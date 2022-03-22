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

#include "hybrid/node_executor/rts/rts_node_task.h"
#include "hybrid/node_executor/rts/rts_task_factory.h"

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/node_utils.h"
#include "common/ge/ge_util.h"
#include "framework/common/op/ge_op_utils.h"

namespace {
constexpr uint8_t kSwitchPredIndex = 0;
constexpr uint8_t kSwitchCompIndex = 1;

const static std::map<rtCondition_t, std::function<bool(int64_t, int64_t)>> kCompHandle = {
  {RT_EQUAL, [](int64_t pred_value, int64_t comp_value) { return pred_value == comp_value; }},
  {RT_NOT_EQUAL, [](int64_t pred_value, int64_t comp_value) { return pred_value != comp_value; }},
  {RT_GREATER, [](int64_t pred_value, int64_t comp_value) { return pred_value > comp_value; }},
  {RT_GREATER_OR_EQUAL, [](int64_t pred_value, int64_t comp_value) { return pred_value >= comp_value; }},
  {RT_LESS, [](int64_t pred_value, int64_t comp_value) { return pred_value < comp_value; }},
  {RT_LESS_OR_EQUAL, [](int64_t pred_value, int64_t comp_value) { return pred_value <= comp_value; }},
};
}

namespace ge {
namespace hybrid {
REGISTER_RTS_TASK_CREATOR(STREAMACTIVE, StreamActiveNodeTask);
REGISTER_RTS_TASK_CREATOR(STREAMSWITCH, StreamSwitchNodeTask);
REGISTER_RTS_TASK_CREATOR(STREAMMERGE, StreamMergeNodeTask);

REGISTER_RTS_TASK_CREATOR(ENTER, PassThroughNodeTask);
REGISTER_RTS_TASK_CREATOR(REFENTER, PassThroughNodeTask);
REGISTER_RTS_TASK_CREATOR(LOOPCOND, PassThroughNodeTask);
REGISTER_RTS_TASK_CREATOR(NEXTITERATION, PassThroughNodeTask);
REGISTER_RTS_TASK_CREATOR(REFNEXTITERATION, PassThroughNodeTask);
REGISTER_RTS_TASK_CREATOR(EXIT, PassThroughNodeTask);
REGISTER_RTS_TASK_CREATOR(REFEXIT, PassThroughNodeTask);

REGISTER_RTS_TASK_CREATOR(LABELSET, LabelSetNodeTask);
REGISTER_RTS_TASK_CREATOR(LABELGOTO, LabelGotoNodeTask);
REGISTER_RTS_TASK_CREATOR(LABELGOTOEX, LabelGotoNodeTask);
REGISTER_RTS_TASK_CREATOR(LABELSWITCH, LabelSwitchNodeTask);
REGISTER_RTS_TASK_CREATOR(LABELSWITCHBYINDEX, LabelSwitchNodeTask);

Status RtsNodeTask::GetScalarIndexValue(TaskContext &task_context, uint32_t index, int64_t &value) {
  auto tensor_value = task_context.GetInput(index);
  GE_CHECK_NOTNULL(tensor_value);
  auto tensor_desc = task_context.MutableInputDesc(index);
  GE_CHECK_NOTNULL(tensor_desc);

  auto data_type = tensor_desc->GetDataType();
  switch (data_type) {
#define CASE_TYPE(DT, VT)                                             \
  case (DT): {                                                        \
    VT data_val{};                                                    \
    GE_CHK_STATUS_RET(tensor_value->CopyScalarValueToHost(data_val)); \
    value = static_cast<int64_t>(data_val);                           \
    break;                                                            \
  }
    // Just accept index data type.
    CASE_TYPE(DT_INT32, int32_t)
    CASE_TYPE(DT_INT64, int64_t)
#undef CASE_TYPE
    default: {
      GELOGE(UNSUPPORTED, "Data type %s not index type.", TypeUtils::DataTypeToSerialString(data_type).c_str());
      return UNSUPPORTED;
    }
  }

  return SUCCESS;
}

Status StreamActiveNodeTask::ExecuteAsync(TaskContext &task_context, std::function<void()> done_callback) {
  GELOGD("[%s] Start to execute.", task_context.GetNodeName());
  const auto &node_state = task_context.GetNodeState();
  node_state->RunStreamActive();
  if (done_callback) {
    GE_CHK_STATUS_RET(task_context.RegisterCallback(done_callback));
  }

  GELOGI("[%s] Done executing successfully.", task_context.GetNodeName());
  return SUCCESS;
}

Status StreamSwitchNodeTask::Init(const HybridModel &model, const NodePtr &node) {
  uint32_t value = 0;
  if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_STREAM_SWITCH_COND, value)) {
    GELOGE(INTERNAL_ERROR, "[%s] Get %s failed.", node->GetName().c_str(), ATTR_NAME_STREAM_SWITCH_COND.c_str());
    return INTERNAL_ERROR;
  }
  rtCondition_t cond = static_cast<rtCondition_t>(value);
  const auto it = kCompHandle.find(cond);
  if (it == kCompHandle.end()) {
    GELOGE(INTERNAL_ERROR, "[%s] Get Condition: %u handle failed.", node->GetName().c_str(), value);
    return INTERNAL_ERROR;
  }

  comp_func_ = it->second;
  GELOGD("[%s] Done initialization successfully, condition is %u.", node->GetName().c_str(), value);
  return SUCCESS;
}

Status StreamSwitchNodeTask::ExecuteAsync(TaskContext &task_context, std::function<void()> done_callback) {
  GELOGD("[%s] Start to execute.", task_context.GetNodeName());
  GE_CHECK_NOTNULL(comp_func_);

  int64_t pred_value = 0;
  GE_CHK_STATUS_RET(GetScalarIndexValue(task_context, kSwitchPredIndex, pred_value));
  int64_t comp_value = 0;
  GE_CHK_STATUS_RET(GetScalarIndexValue(task_context, kSwitchCompIndex, comp_value));

  bool switch_idx = comp_func_(pred_value, comp_value);
  auto node_state = task_context.GetNodeState();
  node_state->SetSwitchIndex(static_cast<int>(switch_idx));

  if (done_callback) {
    GE_CHK_STATUS_RET(task_context.RegisterCallback(done_callback));
  }

  GELOGI("[%s] Done executing successfully, pred value: %ld, comp value: %ld, switch index: %d.",
         task_context.GetNodeName(), pred_value, comp_value, static_cast<int>(switch_idx));
  return SUCCESS;
}

Status StreamMergeNodeTask::ExecuteAsync(TaskContext &task_context, std::function<void()> done_callback) {
  int index = task_context.GetNodeState()->GetMergeIndex();
  GELOGD("[%s] Start to execute, merge index: %d.", task_context.GetNodeName(), index);
  if (index < 0 || index >= task_context.NumInputs()) {
    GELOGE(INTERNAL_ERROR, "[%s] Invalid merge param, inputs num: %d, merge index: %d.",
           task_context.GetNodeName(), task_context.NumInputs(), index);
    return INTERNAL_ERROR;
  }

  const auto in_x = task_context.MutableInput(index); // x
  GE_CHECK_NOTNULL(in_x);
  GE_CHK_STATUS_RET_NOLOG(task_context.SetOutput(MERGE_DATA_OUTPUT, *in_x)); // y

  const auto out_y = task_context.MutableOutput(MERGE_INDEX_OUTPUT);  // value_index
  GE_CHECK_NOTNULL(out_y);
  if (out_y->GetSize() > 0) {
    GE_CHK_RT_RET(rtMemcpyAsync(out_y->MutableData(), out_y->GetSize(), &index, sizeof(index),
                                RT_MEMCPY_HOST_TO_DEVICE_EX, task_context.GetStream()));
  }

  if (done_callback) {
    GE_CHK_STATUS_RET(task_context.RegisterCallback(done_callback));
  }

  task_context.GetNodeState()->SetMergeIndex(-1); // Invalidate for loop.
  GELOGD("[%s] Done executing successfully.", task_context.GetNodeName());
  return SUCCESS;
}

Status PassThroughNodeTask::ExecuteAsync(TaskContext &task_context, std::function<void()> done_callback) {
  GELOGD("[%s] Start to execute.", task_context.GetNodeName());
  const auto in_x = task_context.GetInput(0); // x
  GE_CHECK_NOTNULL(in_x);
  GE_CHK_STATUS_RET_NOLOG(task_context.SetOutput(0, *in_x)); // y

  const auto &node_state = task_context.GetNodeState();
  if (kNextIterationOpTypes.count(node_state->GetType()) > 0) {
    node_state->RunNextIteration();
  }

  if (done_callback) {
    GE_CHK_STATUS_RET(task_context.RegisterCallback(done_callback));
  }

  GELOGD("[%s] Done executing successfully.", task_context.GetNodeName());
  return SUCCESS;
}

Status LabelSetNodeTask::ExecuteAsync(TaskContext &task_context, std::function<void()> done_callback) {
  GELOGD("[%s] Start to execute.", task_context.GetNodeName());

  if (done_callback) {
    GE_CHK_STATUS_RET(task_context.RegisterCallback(done_callback));
  }

  GELOGD("[%s] Done executing successfully.", task_context.GetNodeName());
  return UNSUPPORTED;
}

Status LabelGotoNodeTask::ExecuteAsync(TaskContext &task_context, std::function<void()> done_callback) {
  GELOGD("[%s] Start to execute.", task_context.GetNodeName());

  if (done_callback) {
    GE_CHK_STATUS_RET(task_context.RegisterCallback(done_callback));
  }

  GELOGD("[%s] Done executing successfully.", task_context.GetNodeName());
  return UNSUPPORTED;
}

Status LabelSwitchNodeTask::ExecuteAsync(TaskContext &task_context, std::function<void()> done_callback) {
  GELOGD("[%s] Start to execute.", task_context.GetNodeName());

  if (done_callback) {
    GE_CHK_STATUS_RET(task_context.RegisterCallback(done_callback));
  }

  GELOGD("[%s] Done executing successfully.", task_context.GetNodeName());
  return UNSUPPORTED;
}
}  // namespace hybrid
}  // namespace ge