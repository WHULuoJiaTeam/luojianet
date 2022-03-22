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

#include "graph/load/model_manager/task_info/kernel_ex_task_info.h"

#include <vector>

#include "cce/aicpu_engine_struct.h"
#include "common/ge/ge_util.h"
#include "common/properties_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/fmk_error_codes.h"
#include "external/graph/attr_value.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_manager.h"
#include "framework/common/debug/log.h"
#include "runtime/rt.h"

namespace {
const char *const kAicpuAllshape = "_AllShape";
}  // namespace

namespace ge {
Status KernelExTaskInfo::InitTaskExtInfo(const std::string &ext_info, const OpDescPtr &op_desc) {
  if (ext_info.empty()) {
    return SUCCESS;
  }
  int32_t unknown_shape_type_val = 0;
  (void) AttrUtils::GetInt(op_desc, ::ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
  UnknowShapeOpType unknown_type = static_cast<UnknowShapeOpType>(unknown_shape_type_val);
  uint32_t num_inputs = op_desc->GetInputsSize();
  uint32_t num_outputs = op_desc->GetOutputsSize();
  std::shared_ptr<ge::hybrid::AicpuExtInfoHandler> ext_handle(
          new(std::nothrow) ::ge::hybrid::AicpuExtInfoHandler(op_desc->GetName(),
                                                              num_inputs,
                                                              num_outputs,
                                                              unknown_type));
  GE_CHK_BOOL_RET_STATUS(ext_handle != nullptr, FAILED, "[Malloc][Memory] for aicpu_ext_handle failed!");
  GE_CHK_STATUS_RET(ext_handle->Parse(ext_info),
                    "[Parse][KernelExtInfo] failed, kernel_ext_info_size=%zu.", ext_info.size());
  GE_CHK_STATUS_RET(ext_handle->UpdateExecuteMode(true), "[Update][ExecuteMode] failed.");
  GELOGD("Update aicpu_task ext_info bit_map execute mode to 1.");
  topic_type_flag_ = ext_handle->GetTopicTypeFlag();

  bool all_shape = false;
  (void)AttrUtils::GetBool(op_desc, kAicpuAllshape, all_shape);
  if (all_shape) {
    GELOGD("Aicpu all_shape kernel need to update io shape.");
    for (uint32_t i = 0; i < num_inputs; i++) {
      auto input_desc = op_desc->MutableInputDesc(i);
      GE_CHECK_NOTNULL(input_desc);
      GE_CHK_STATUS_RET(ext_handle->UpdateInputShapeAndType(i, *input_desc),
                        "[Call][UpdateInputShapeAndType] Input[%u] update input shape failed, op:%s.",
                        i, op_desc->GetName().c_str());
    }
    if (unknown_type != DEPEND_COMPUTE) {
      for (uint32_t j = 0; j < num_outputs; j++) {
        auto output_desc = op_desc->MutableOutputDesc(j);
        GE_CHECK_NOTNULL(output_desc);
        GE_CHK_STATUS_RET(ext_handle->UpdateOutputShapeAndType(j, *output_desc),
                          "[Call][UpdateOutputShapeAndType] Output[%u] update output shape failed, op:%s.",
                          j, op_desc->GetName().c_str());
      }
    }
  }

  AttrUtils::GetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, is_blocking_aicpu_op_);
  GELOGD("Get op:%s attribute(is_blocking_op), value:%d", op_desc->GetName().c_str(), is_blocking_aicpu_op_);

  if (UpdateEventIdForAicpuBlockingOp(op_desc, ext_handle) != SUCCESS) {
    GELOGE(FAILED, "[Call][UpdateEventIdForAicpuBlockingOp] failed for op:%s(%s)",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }

  auto rt_ret = rtMalloc(&ext_info_addr_, ext_handle->GetExtInfoLen(), RT_MEMORY_HBM);
  GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                  REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%zu, ret:0x%X", ext_info.size(), rt_ret);
                  GELOGE(RT_FAILED, "[RtMalloc][ExtInfo] error:0x%X, size=%zu", rt_ret, ext_info.size());
                  return RT_ERROR_TO_GE_STATUS(rt_ret);)
  rt_ret = rtMemcpy(ext_info_addr_, ext_handle->GetExtInfoLen(), ext_handle->GetExtInfo(),
                    ext_handle->GetExtInfoLen(), RT_MEMCPY_HOST_TO_DEVICE);
  GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                  REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%zu, ret:0x%X",
                                    ext_handle->GetExtInfoLen(), rt_ret);
                  GELOGE(RT_FAILED, "[RtMemcpy][ExtInfo] error:0x%X, size=%zu", rt_ret, ext_info.size());
                  return RT_ERROR_TO_GE_STATUS(rt_ret);)
  return SUCCESS;
}

Status KernelExTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("KernelExTaskInfo Init Start.");
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model_ = davinci_model;
  Status ret = SetStream(task_def.stream_id(), davinci_model_->GetStreamList());
  if (ret != SUCCESS) {
    return ret;
  }

  auto kernel_ex_def = task_def.kernel_ex();
  const RuntimeParam &rts_param = davinci_model_->GetRuntimeParam();

  // 1. Copy context from kernelExDef.private to workspace
  uint32_t op_index = kernel_ex_def.op_index();
  OpDescPtr op_desc = davinci_model_->GetOpByIndex(op_index);
  op_desc_ = op_desc;
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "Can't get op_desc from davinci_model by index:%u", op_index);
    GELOGE(INTERNAL_ERROR, "[Get][Op] by index failed, index:%u is out of range!", op_index);
    return INTERNAL_ERROR;
  }

  // 2. Reconstruct kernelExDef.args to STR_FWK_OP_KERNEL
  STR_FWK_OP_KERNEL fwk_op_kernel = {0};
  if (sizeof(STR_FWK_OP_KERNEL) < kernel_ex_def.args_size()) {
    REPORT_INNER_ERROR("E19999", "Param kernel_ex_def.args_size():%u > sizeof(STR_FWK_OP_KERNEL):%zu, "
                       "check invalid", kernel_ex_def.args_size(), sizeof(STR_FWK_OP_KERNEL));
    GELOGE(FAILED, "[Check][Param] sizeof STR_FWK_OP_KERNEL is: %zu, but args_size is: %u", sizeof(STR_FWK_OP_KERNEL),
           kernel_ex_def.args_size());
    return FAILED;
  }
  errno_t sec_ret =
      memcpy_s(&fwk_op_kernel, sizeof(STR_FWK_OP_KERNEL), kernel_ex_def.args().data(), kernel_ex_def.args_size());
  if (sec_ret != EOK) {
    REPORT_CALL_ERROR("E19999", "Call memcpy_s fail, size:%zu, ret:0x%X", sizeof(STR_FWK_OP_KERNEL), sec_ret);
    GELOGE(FAILED, "[Call][Memcpy] failed, size:%zu, ret: %d", sizeof(STR_FWK_OP_KERNEL), sec_ret);
    return FAILED;
  }

  const auto &ext_info = kernel_ex_def.kernel_ext_info();
  GE_CHK_STATUS_RET(InitTaskExtInfo(ext_info, op_desc),
                    "[Init][TaskExtInfo] failed, ext_info size=%zu, op:%s",
                    ext_info.size(), op_desc->GetName().c_str());

  GELOGI("Node[%s] type[%s] kernel_ext_info size=%zu, ext_info_addr_=%p", op_desc->GetName().c_str(),
         op_desc->GetType().c_str(), ext_info.size(), ext_info_addr_);

  // 2.1 get loop cond variable for tensor array write
  uint64_t step_id_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(davinci_model_->GetGlobalStep()));

  auto session_id = davinci_model_->GetSessionId();
  fwk_op_kernel.fwkKernelBase.fwk_kernel.sessionID = session_id;

  // 2.2 Collect aicpu kernel
  uint64_t kernel_id = fwk_op_kernel.fwkKernelBase.fwk_kernel.kernelID;
  GE_IF_BOOL_EXEC(ModelManager::GetInstance()->CreateAicpuKernel(session_id, davinci_model->Id(),
                                                                 davinci_model->SubModelId(), kernel_id) != SUCCESS,
                  REPORT_CALL_ERROR("E19999", "CreateAicpuKernel fail, session_id:%lu, model_id:%u, kernel_id:%lu",
                                    session_id, davinci_model->Id(), kernel_id);
                  GELOGE(FAILED, "[Create][AicpuKernel] fail, session_id:%lu, model_id:%u, kernel_id:%lu",
                         session_id, davinci_model->Id(), kernel_id);
                  return FAILED;)
  // 2.3 Create session
  GE_CHECK_NOTNULL(ModelManager::GetInstance());
  ret = ModelManager::GetInstance()->CreateAicpuSession(session_id);
  GE_IF_BOOL_EXEC(ret != SUCCESS,
                  REPORT_CALL_ERROR("E19999", "CreateAicpuSession fail, session_id:%lu", session_id);
                  GELOGE(ret, "[Create][AicpuSession] error. session id:%lu", session_id);
                  return ret;)

  kernel_buf_size_ = sizeof(STR_FWK_OP_KERNEL);
  if (davinci_model_->IsKnownNode()) {
    void *input_output_addr = davinci_model_->GetCurrentArgsAddr(args_offset_);
    fwk_op_kernel.fwkKernelBase.fwk_kernel.inputOutputAddr =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(input_output_addr));
    void *workspace_base_addr = nullptr;
    rtError_t rt_ret = rtMalloc(&workspace_base_addr, kernel_ex_def.task_info_size(), RT_MEMORY_HBM);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%u, ret:0x%X",
                                      kernel_ex_def.task_info_size(), rt_ret);
                    GELOGE(RT_FAILED, "[Call][RtMalloc] failed, size:%u, ret:0x%X",
                           kernel_ex_def.task_info_size(), rt_ret);
                    return RT_ERROR_TO_GE_STATUS(rt_ret););
    rt_ret = rtMemcpy(workspace_base_addr, kernel_ex_def.task_info_size(), kernel_ex_def.task_info().data(),
                      kernel_ex_def.task_info_size(), RT_MEMCPY_HOST_TO_DEVICE);
    fwk_op_kernel.fwkKernelBase.fwk_kernel.workspaceBaseAddr =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(workspace_base_addr));
    fwk_op_kernel.fwkKernelBase.fwk_kernel.stepIDAddr = step_id_addr;
    fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoLen = ext_info.size();
    fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoAddr = reinterpret_cast<uintptr_t>(ext_info_addr_);

    rt_ret = rtMalloc(&kernel_buf_, kernel_buf_size_, RT_MEMORY_HBM);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, ret:0x%X, size:%u", rt_ret, kernel_buf_size_);
                    GELOGE(RT_FAILED, "[Call][RtMalloc] failed, ret:0x%X, size:%u", rt_ret, kernel_buf_size_);
                    return RT_ERROR_TO_GE_STATUS(rt_ret);)

    rt_ret = rtMemcpy(kernel_buf_, kernel_buf_size_, static_cast<void *>(&fwk_op_kernel), kernel_buf_size_,
                      RT_MEMCPY_HOST_TO_DEVICE);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, ret:0x%X, size:%u", rt_ret, kernel_buf_size_);
                    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, ret:0x%X, size:%u", rt_ret, kernel_buf_size_);
                    return RT_ERROR_TO_GE_STATUS(rt_ret);)

    SetIoAddrs(op_desc);
    InitDumpFlag(op_desc);
    InitDumpArgs(input_output_addr, op_desc);
    GELOGI("KernelExTaskInfo knonw node Init Success.");
    return SUCCESS;
  }

  // 3. Set workspaceaddr, inputOutputDataAddr
  Status ge_ret = CopyTaskInfo(kernel_ex_def, rts_param, op_desc);
  if (ge_ret != SUCCESS) {
    GELOGE(ge_ret, "[Copy][TaskInfo] to workspace failed, op:%s.", op_desc->GetName().c_str());
    return ge_ret;
  }

  const vector<void *> workspace_data_addrs = ModelUtils::GetWorkspaceDataAddrs(rts_param, op_desc);
  if (workspace_data_addrs.empty()) {
    REPORT_CALL_ERROR("E19999", "workspace_data_addrs is empty in op:%s(%s), check invalid",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Check][Param] workspace_data_addrs is empty in op:%s(%s).",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }

  uint64_t workspace_base_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(workspace_data_addrs[0]));
  const vector<void *> input_addrs = ModelUtils::GetInputDataAddrs(rts_param, op_desc);
  const vector<void *> output_addrs = ModelUtils::GetOutputDataAddrs(rts_param, op_desc);
  vector<void *> io_addrs;
  io_addrs.insert(io_addrs.end(), input_addrs.begin(), input_addrs.end());
  io_addrs.insert(io_addrs.end(), output_addrs.begin(), output_addrs.end());

  auto addrs_size = sizeof(uint64_t) * (io_addrs.size());
  if (addrs_size > 0) {
    rtError_t rt_ret = rtMalloc(&input_output_addr_, addrs_size, RT_MEMORY_HBM);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, ret:0x%X, size:%lu", rt_ret, addrs_size);
                    GELOGE(RT_FAILED, "[Call][RtMalloc] failed, ret:0x%X, size:%lu", rt_ret, addrs_size);
                    return RT_ERROR_TO_GE_STATUS(rt_ret);)

    rt_ret = rtMemcpy(input_output_addr_, addrs_size, io_addrs.data(), addrs_size, RT_MEMCPY_HOST_TO_DEVICE);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, ret:0x%X, size:%lu", rt_ret, addrs_size);
                    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, ret:0x%X, size:%lu", rt_ret, addrs_size);
                    return RT_ERROR_TO_GE_STATUS(rt_ret);)

    InitDumpFlag(op_desc);
    InitDumpArgs(input_output_addr_, op_desc);
  }

  uint64_t input_output_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(input_output_addr_));
  fwk_op_kernel.fwkKernelBase.fwk_kernel.workspaceBaseAddr = workspace_base_addr;
  fwk_op_kernel.fwkKernelBase.fwk_kernel.inputOutputAddr = input_output_addr;
  fwk_op_kernel.fwkKernelBase.fwk_kernel.stepIDAddr = step_id_addr;
  fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoLen = ext_info.size();
  fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoAddr = reinterpret_cast<uintptr_t>(ext_info_addr_);

  // 4. Return result
  rtError_t rt_ret = rtMalloc(&kernel_buf_, sizeof(STR_FWK_OP_KERNEL), RT_MEMORY_HBM);
  GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                  REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, ret:0x%X, size:%zu",
                                    rt_ret, sizeof(STR_FWK_OP_KERNEL));
                  GELOGE(RT_FAILED, "[Call][RtMalloc] failed, ret:0x%X, size:%zu", rt_ret, sizeof(STR_FWK_OP_KERNEL));
                  return RT_ERROR_TO_GE_STATUS(rt_ret);)

  rt_ret = rtMemcpy(kernel_buf_, sizeof(STR_FWK_OP_KERNEL), static_cast<void *>(&fwk_op_kernel),
                    sizeof(STR_FWK_OP_KERNEL), RT_MEMCPY_HOST_TO_DEVICE);
  GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                  REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, ret:0x%X, size:%zu",
                                    rt_ret, sizeof(STR_FWK_OP_KERNEL));
                  GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, ret:0x%X, size:%zu", rt_ret, sizeof(STR_FWK_OP_KERNEL));
                  return RT_ERROR_TO_GE_STATUS(rt_ret);)

  davinci_model_->SetZeroCopyAddr(op_desc, io_addrs, io_addrs.data(), input_output_addr_, addrs_size, 0);
  SetIoAddrs(op_desc);
  GELOGI("KernelExTaskInfo Init Success. session id: %lu", session_id);
  return SUCCESS;
}

void KernelExTaskInfo::InitDumpFlag(const OpDescPtr &op_desc) {
  if (davinci_model_->OpNeedDump(op_desc->GetName())) {
    GELOGD("Op %s need init dump flag in kernel ex task info", op_desc->GetName().c_str());
    dump_flag_ = RT_KERNEL_DUMPFLAG;
  }
}

void KernelExTaskInfo::InitDumpArgs(void *addr, const OpDescPtr &op_desc) {
  if (davinci_model_->OpNeedDump(op_desc->GetName())) {
    GELOGD("Op %s need dump in kernel ex task info", op_desc->GetName().c_str());
    dump_args_ = addr;
  }
  if (davinci_model_->GetOpDugReg()) {
    GELOGD("Op debug is open in kernel ex task info");
    dump_args_ = addr;
  }
}

Status KernelExTaskInfo::CalculateArgs(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  auto kernel_ex_def = task_def.kernel_ex();
  uint32_t op_index = kernel_ex_def.op_index();
  OpDescPtr op_desc = davinci_model->GetOpByIndex(op_index);
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "Can't get op_desc from davinci_model by index:%u", op_index);
    GELOGE(INTERNAL_ERROR, "[Get][Op] By Index, index:%u is out of range!", op_index);
    return INTERNAL_ERROR;
  }
  args_offset_ = davinci_model->GetTotalArgsSize();
  const size_t inputs_size = op_desc->GetInputsSize();
  const size_t outputs_size = op_desc->GetOutputsSize();
  // aicpu kernel input/output size
  size_t mem_length = inputs_size + outputs_size;
  uint32_t mem_size = sizeof(uint64_t) * mem_length;
  davinci_model->SetTotalArgsSize(mem_size);
  GELOGI("kernel task name %s, args_size %u, args_offset %u", op_desc->GetName().c_str(), mem_size, args_offset_);

  // alloc fixed addr
  string peer_input_name;
  if (AttrUtils::GetStr(op_desc, ATTR_DYNAMIC_SHAPE_FIXED_ADDR, peer_input_name) && !peer_input_name.empty()) {
    uint32_t output_index = davinci_model->GetFixedAddrOutputIndex(peer_input_name);
    if (output_index > outputs_size) {
      REPORT_INNER_ERROR("E19999", "The output size[%zu] and output index[%u] in op:%s(%s) are inconsistent, "
                         "check invalid", outputs_size, output_index,
                         op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(FAILED, "[Check][Param] The output size[%zu] and output index[%u] in op:%s(%s) are inconsistent.",
             outputs_size, output_index, op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return FAILED;
    }
    fixed_addr_offset_ = davinci_model->GetFixedAddrsSize(peer_input_name);
    auto tensor_desc = op_desc->GetOutputDesc(output_index);
    int64_t tensor_size = 0;
    GE_CHK_STATUS(TensorUtils::GetSize(tensor_desc, tensor_size));
    davinci_model->SetTotalFixedAddrsSize(peer_input_name, tensor_size);
    GELOGI("Calculate stream switch task args , tensor size is %ld, fixed addr offset %ld", tensor_size,
           fixed_addr_offset_);
  }
  return SUCCESS;
}

void KernelExTaskInfo::SetIoAddrs(const OpDescPtr &op_desc) {
  const RuntimeParam &rts_param = davinci_model_->GetRuntimeParam();
  vector<void *> input_data_addrs = ModelUtils::GetInputDataAddrs(rts_param, op_desc);
  vector<void *> output_data_addrs = ModelUtils::GetOutputDataAddrs(rts_param, op_desc);
  if (!op_desc->HasAttr(ATTR_DYNAMIC_SHAPE_FIXED_ADDR)) {
    io_addrs_.insert(io_addrs_.end(), input_data_addrs.begin(), input_data_addrs.end());
    io_addrs_.insert(io_addrs_.end(), output_data_addrs.begin(), output_data_addrs.end());
  } else {
    string peer_input_name;
    if (AttrUtils::GetStr(op_desc, ATTR_DYNAMIC_SHAPE_FIXED_ADDR, peer_input_name)) {
      uint32_t output_index = davinci_model_->GetFixedAddrOutputIndex(peer_input_name);
      if (output_index > output_data_addrs.size()) {
        REPORT_INNER_ERROR("E19999", "The output data addr size[%zu] and output index[%u] in op:%s(%s) "
                           "are inconsistent, check invalid", output_data_addrs.size(), output_index,
                           op_desc->GetName().c_str(), op_desc->GetType().c_str());
        GELOGE(FAILED, "[Check][Param] The output data addr size[%zu] and output index[%u] in op:%s(%s) "
               "are inconsistent.", output_data_addrs.size(), output_index,
               op_desc->GetName().c_str(), op_desc->GetType().c_str());
        return;
      }
      io_addrs_.insert(io_addrs_.end(), input_data_addrs.begin(), input_data_addrs.end());
      for (size_t i = 0; i < output_data_addrs.size(); ++i) {
        if (i == output_index) {
          void *fixed_addr = davinci_model_->GetCurrentFixedAddr(fixed_addr_offset_);
          io_addrs_.emplace_back(fixed_addr);
          continue;
        }
        io_addrs_.emplace_back(output_data_addrs[i]);
      }
    }
  }
}

Status KernelExTaskInfo::UpdateArgs() {
  GELOGI("KernelExTaskInfo::UpdateArgs in.");
  davinci_model_->SetTotalIOAddrs(io_addrs_);
  davinci_model_->UpdateOpIOAddrs(task_id_, stream_id_, io_addrs_);
  GELOGI("KernelExTaskInfo::UpdateArgs success.");
  return SUCCESS;
}

Status KernelExTaskInfo::CopyTaskInfo(const domi::KernelExDef &kernel_def, const RuntimeParam &rts_param,
                                      const OpDescPtr &op_desc) {
  // Userspace copy need virtual address.
  const vector<int64_t> workspace_data_sizes = ModelUtils::GetWorkspaceSize(op_desc);
  const vector<void *> workspace_data_addrs = ModelUtils::GetWorkspaceDataAddrs(rts_param, op_desc);
  if (workspace_data_addrs.empty() || workspace_data_sizes.empty()) {
    REPORT_INNER_ERROR("E19999", "Node:%s(%s) workspace addr:%zu or size:%zu empty, check invalid",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                       workspace_data_addrs.size(), workspace_data_sizes.size());
    GELOGE(FAILED, "[Check][Param] Node:%s invalid workspace, addrs is %zu, size is %zu.", op_desc->GetName().c_str(),
           workspace_data_addrs.size(), workspace_data_sizes.size());
    return FAILED;
  }

  if (workspace_data_addrs[0] == nullptr) {
    REPORT_INNER_ERROR("E19999", "Node:%s(%s) workspace addr is nullptr, check invalid",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Check][Param] Node:%s workspace addrs is null.", op_desc->GetName().c_str());
    return FAILED;
  }

  if (workspace_data_sizes[0] < static_cast<int64_t>(kernel_def.task_info_size())) {
    REPORT_INNER_ERROR("E19999", "Node:%s(%s) workspace size:%ld < task info size:%d, check invalid",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                       workspace_data_sizes[0], kernel_def.task_info_size());
    GELOGE(FAILED, "[Check][Param] Node:%s workspace size is %ld, task info size is %d.", op_desc->GetName().c_str(),
           workspace_data_sizes[0], kernel_def.task_info_size());
    return FAILED;
  }

  rtError_t rt_ret = rtMemcpy(workspace_data_addrs[0], kernel_def.task_info_size(), kernel_def.task_info().data(),
                              kernel_def.task_info_size(), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, ret:0x%X, size:%d", rt_ret, kernel_def.task_info_size());
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, ret:0x%X, size:%d", rt_ret, kernel_def.task_info_size());
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  return SUCCESS;
}

Status KernelExTaskInfo::Distribute() {
  GELOGI("KernelExTaskInfo Distribute Start.");
  // Use the fifth and sixth bits of dump_flag_ indicate the value of topic_type.
  // xxxxxxxx xxxxxxxx xxxxxxxx xx00xxxx: DEVICE_ONLY
  // xxxxxxxx xxxxxxxx xxxxxxxx xx01xxxx: DEVICE_FIRST
  // xxxxxxxx xxxxxxxx xxxxxxxx xx10xxxx: HOST_ONLY
  // xxxxxxxx xxxxxxxx xxxxxxxx xx11xxxx: HOST_FIRST
  if (topic_type_flag_ > 0) {
    dump_flag_ = dump_flag_ | static_cast<uint32_t>(topic_type_flag_);
  }
  rtError_t rt_ret = rtKernelLaunchFwk(op_desc_->GetName().c_str(), kernel_buf_, kernel_buf_size_, dump_flag_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtKernelLaunchEx failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtKernelLaunchEx] failed, ret:0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  if (davinci_model_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "Check param davinci_model nullptr");
    GELOGE(PARAM_INVALID, "[Check][Param] davinci_model_ is null.");
    return PARAM_INVALID;
  }

  uint32_t task_id = 0;
  uint32_t stream_id = 0;  //  for profiling
  rt_ret = rtModelGetTaskId(davinci_model_->GetRtModelHandle(), &task_id, &stream_id);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtModelGetTaskId failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtModelGetTaskId] failed, ret:0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  task_id_ = task_id;
  stream_id_ = stream_id;

  GELOGI("KernelExTaskInfo Distribute Success. task id: %u, stream id: %u", task_id_, stream_id_);
  if (is_blocking_aicpu_op_) {
    if (DistributeWaitTaskForAicpuBlockingOp() != SUCCESS) {
      GELOGE(FAILED, "[Call][DistributeWaitTaskForAicpuBlockingOp] Call DistributeWaitTaskForAicpuBlockingOp failed");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status KernelExTaskInfo::CheckDeviceSupportBlockingAicpuOpProcess(bool &is_support) {
  int32_t device_id = 0;
  auto rt_ret = rtGetDevice(&device_id);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtGetDevice failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][rtGetDevice] failed, ret:0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  int32_t value = 0;
  rt_ret = rtGetDeviceCapability(device_id, FEATURE_TYPE_BLOCKING_OPERATOR, RT_MODULE_TYPE_AICPU, &value);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtGetDeviceCapability failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][rtGetDeviceCapability] failed, ret:0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  if (value != RT_AICPU_BLOCKING_OP_NOT_SUPPORT && value != RT_AICPU_BLOCKING_OP_SUPPORT) {
    REPORT_INNER_ERROR("E19999", "Value should be %d or %d but %d",
                       RT_AICPU_BLOCKING_OP_NOT_SUPPORT, RT_AICPU_BLOCKING_OP_SUPPORT, value);
    GELOGE(FAILED, "[Check][Value] Value should be %d or %d but %d",
           RT_AICPU_BLOCKING_OP_NOT_SUPPORT, RT_AICPU_BLOCKING_OP_SUPPORT, value);
    return FAILED;
  }
  is_support = (value == RT_AICPU_BLOCKING_OP_SUPPORT ? true : false);
  return SUCCESS;
}

Status KernelExTaskInfo::UpdateEventIdForAicpuBlockingOp(const OpDescPtr &op_desc,
    std::shared_ptr<ge::hybrid::AicpuExtInfoHandler> &ext_handle) {
  if (is_blocking_aicpu_op_) {
    bool is_support = false;
    if (CheckDeviceSupportBlockingAicpuOpProcess(is_support) != SUCCESS) {
      GELOGE(FAILED, "[Call][CheckDeviceSupportBlockingAicpuOpProcess] Call CheckDeviceSupportBlockingAicpuOpProcess failed");
      return FAILED;
    }
    if (!is_support) {
      GELOGD("Device not support blocking aicpu op process");
      return SUCCESS;
    }
    uint32_t event_id = 0;
    if (davinci_model_->GetEventIdForBlockingAicpuOp(op_desc, stream_, event_id) != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Get event id failed for op:%s(%s).", op_desc->GetName().c_str(),
                        op_desc->GetType().c_str());
      GELOGE(FAILED, "[Get][EventId] Get event id failed for op:%s(%s)", op_desc->GetName().c_str(),
             op_desc->GetType().c_str());
      return FAILED;
    }
    if (ext_handle->UpdateEventId(event_id) != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Update event id failed for op:%s(%s).", op_desc->GetName().c_str(),
                        op_desc->GetType().c_str());
      GELOGE(FAILED, "[Update][EventId] Update event id failed for op:%s(%s)", op_desc->GetName().c_str(),
             op_desc->GetType().c_str());
      return FAILED;
    }
    GELOGI("Update event_id=%u success", event_id);
  }
  return SUCCESS;
}

Status KernelExTaskInfo::DistributeWaitTaskForAicpuBlockingOp() {
  bool is_support = false;
  if (CheckDeviceSupportBlockingAicpuOpProcess(is_support) != SUCCESS) {
    GELOGE(FAILED, "[Call][CheckDeviceSupportBlockingAicpuOpProcess] Call CheckDeviceSupportBlockingAicpuOpProcess failed");
    return FAILED;
  }
  if (!is_support) {
    GELOGD("Device not support blocking aicpu op process.");
    return SUCCESS;
  }
  GELOGD("Distribute wait task begin");
  rtEvent_t rt_event = nullptr;
  if (davinci_model_->GetEventByStream(stream_, rt_event) != SUCCESS) {
    GELOGE(FAILED, "[Call][GetEventByStream] Call GetEventByStream failed");
    return FAILED;
  }
  auto rt_ret = rtStreamWaitEvent(stream_, rt_event);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtStreamWaitEvent failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtApi] failed, ret:0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  rt_ret = rtEventReset(rt_event, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtEventReset failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtApi] failed, ret:0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  return SUCCESS;
}

Status KernelExTaskInfo::Release() {
  Status ret = SUCCESS;
  if (kernel_buf_ != nullptr) {
    rtError_t rt_ret = rtFree(kernel_buf_);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("rtFree error, ret: 0x%X", rt_ret);
      ret = RT_ERROR_TO_GE_STATUS(rt_ret);
    } else {
      kernel_buf_ = nullptr;
    }
  }
  if (input_output_addr_ != nullptr) {
    rtError_t rt_ret = rtFree(input_output_addr_);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("rtFree error, ret: 0x%X", rt_ret);
      ret = RT_ERROR_TO_GE_STATUS(rt_ret);
    } else {
      input_output_addr_ = nullptr;
    }
  }
  if (ext_info_addr_ != nullptr) {
    rtError_t rt_ret = rtFree(ext_info_addr_);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("rtFree ext_info_addr[%p] error, ret: 0x%X", ext_info_addr_, rt_ret);
      ret = RT_ERROR_TO_GE_STATUS(rt_ret);
    } else {
      ext_info_addr_ = nullptr;
    }
  }
  return ret;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_KERNEL_EX, KernelExTaskInfo);
}  // namespace ge
