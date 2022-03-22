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

#include "graph/load/model_manager/task_info/hccl_task_info.h"

#include <utility>

#include "common/opskernel/ops_kernel_info_store.h"
#include "framework/common/debug/ge_log.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_utils.h"

namespace ge {
std::mutex HcclTaskInfo::hccl_follow_stream_mutex_;

HcclTaskInfo::~HcclTaskInfo() {
  if (private_def_ != nullptr) {
    rtError_t ret = rtFreeHost(private_def_);
    if (ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtFreeHost failed, ret:0x%X", ret);
      GELOGE(RT_FAILED, "[Call][RtFree] Fail, ret = 0x%X.", ret);
    }
    private_def_ = nullptr;
  }
  davinci_model_ = nullptr;
  ops_kernel_store_ = nullptr;
  args_ = nullptr;
}
Status HcclTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("HcclTaskInfo Init Start.");
  if (davinci_model == nullptr) {
    REPORT_INNER_ERROR("E19999", "Check param davinci_model nullptr");
    GELOGE(PARAM_INVALID, "[Check][Param] davinci_model is null!");
    return PARAM_INVALID;
  }
  davinci_model_ = davinci_model;
  Status ret = SetStream(task_def.stream_id(), davinci_model->GetStreamList());
  if (ret != SUCCESS) {
    return ret;
  }
  GetPrivateDefByTaskDef(task_def);
  auto hccl_def = task_def.kernel_hccl();
  uint32_t op_index = hccl_def.op_index();
  GELOGI("HcclTaskInfo Init, op_index is: %u", op_index);

  // Get HCCL op
  const auto op_desc = davinci_model_->GetOpByIndex(op_index);
  GE_CHECK_NOTNULL(op_desc);

  // Create the kernel hccl infos
  CreateKernelHcclInfo(op_desc);

  // Initialize the hccl_type of all kernel hccl info
  HcomOmeUtil::GetHcclType(task_def, kernel_hccl_infos_);

  // Only in Horovod scenario should get the inputName and GeShape
  ret = HcomOmeUtil::GetHorovodInputs(op_desc, kernel_hccl_infos_);
  if (ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Call GetHorovodInputs fail for op:%s(%s)",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(ret, "[Get][HorovodInputs] fail for op:%s(%s)", op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return ret;
  }
  Status dmrt = HcomOmeUtil::GetHcclDataType(op_desc, kernel_hccl_infos_);
  if (dmrt != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Call GetHcclDataType fail for op:%s(%s)",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(dmrt, "[Get][HcomDataType] fail for op:%s(%s)", op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return dmrt;
  }
  dmrt = HcomOmeUtil::GetHcclCount(op_desc, kernel_hccl_infos_);
  if (dmrt != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Call GetHcclCount fail for op:%s(%s)",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(dmrt, "[Get][HcomCount] fail for op:%s(%s)", op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return dmrt;
  }
  // Only HCOMBROADCAST and HVDCALLBACKBROADCAST need to get the rootId
  dmrt = HcomOmeUtil::GetAllRootId(op_desc, kernel_hccl_infos_);
  if (dmrt != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Call GetAllRootId fail for op:%s(%s)",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(dmrt, "[Get][RootId] fail for op:%s(%s)", op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return dmrt;
  }

  // GE's new process: hccl declares the number of streams required, creates a stream by GE, and sends it to hccl
  ret = SetFollowStream(op_desc, davinci_model);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][Stream] Fail for op:%s(%s)", op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return ret;
  }

  if (davinci_model_->IsKnownNode()) {
    args_ = davinci_model_->GetCurrentArgsAddr(args_offset_);
    GELOGI("Known node %s args addr %p, offset %u.", op_desc->GetName().c_str(), args_, args_offset_);
  }

  ret = SetAddrs(op_desc, kernel_hccl_infos_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][Addrs] Fail for op:%s(%s)", op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return ret;
  }
  // GE's new process: hccl declares the need for Workspace size, and GE allocates Workspace
  ret = SetWorkspace(op_desc, kernel_hccl_infos_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][Workspace] Fail for op:%s(%s)", op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return ret;
  }

  SetIoAddrs(op_desc);
  GELOGI("HcclTaskInfo Init Success");
  return SUCCESS;
}

Status HcclTaskInfo::SetFollowStream(const ge::ConstOpDescPtr &op_desc, DavinciModel *davinci_model) {
  if (!HcomOmeUtil::IsHCOMOp(op_desc->GetType())) {
    GELOGI("Node %s Optye %s no need to create slave streams.", op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return SUCCESS;
  }
  Status ret;
  int64_t hccl_stream_num = 0;
  if (!ge::AttrUtils::GetInt(op_desc, "used_stream_num", hccl_stream_num)) {
    GELOGI("op_desc has no attr used_stream_num!");
  }

  std::lock_guard<std::mutex> lock(hccl_follow_stream_mutex_);
  int64_t main_stream_id = op_desc->GetStreamId();
  const std::map<int64_t, std::vector<rtStream_t>> &main_follow_stream_mapping = davinci_model->GetHcclFolowStream();

  if (main_follow_stream_mapping.find(main_stream_id) != main_follow_stream_mapping.end()) {
    const std::vector<rtStream_t> &follow_stream_usage = main_follow_stream_mapping.at(main_stream_id);
    if (static_cast<size_t>(hccl_stream_num) <= follow_stream_usage.size()) {
      GELOGI("capacity of follow stream is enough to be reused.");
      for (int64_t i = 0; i < hccl_stream_num; i++) {
        hccl_stream_list_.emplace_back(follow_stream_usage.at(i));
      }
    } else {
      GELOGI("need to reuse follow stream and create new follow stream.");
      size_t created_stream_num = follow_stream_usage.size();
      for (const auto &stream : follow_stream_usage) {
        hccl_stream_list_.emplace_back(stream);
      }
      ret = CreateStream(hccl_stream_num - created_stream_num, davinci_model, main_stream_id);
      if (ret != SUCCESS) {
        GELOGE(RT_FAILED, "[Create][Stream] for %s failed, stream id:%ld, stream num:%ld.",
               op_desc->GetName().c_str(), main_stream_id, hccl_stream_num - created_stream_num);
        return RT_ERROR_TO_GE_STATUS(ret);
      }
    }
    GELOGI("Initialize hccl slave stream success, hcclStreamNum =%ld", hccl_stream_num);
  } else {
    GELOGI("need to create follow stream for %s with new mainstream %ld.", op_desc->GetName().c_str(), main_stream_id);
    ret = CreateStream(hccl_stream_num, davinci_model, main_stream_id);
    if (ret != SUCCESS) {
      GELOGE(RT_FAILED, "[Create][Stream] for %s failed, stream id:%ld, stream num:%ld.",
             op_desc->GetName().c_str(), main_stream_id, hccl_stream_num);
      return RT_ERROR_TO_GE_STATUS(ret);
    }
  }
  return SUCCESS;
}

Status HcclTaskInfo::CreateStream(int64_t stream_num, DavinciModel *davinci_model, int64_t main_stream_id) {
  GELOGI("Start to create %ld hccl stream.", stream_num);
  for (int64_t i = 0; i < stream_num; ++i) {
    rtStream_t stream = nullptr;
    rtError_t rt_ret =
        rtStreamCreateWithFlags(&stream, davinci_model->Priority(), RT_STREAM_PERSISTENT | RT_STREAM_FORCE_COPY);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtStreamCreateWithFlags failed, ret:0x%X, stream_idx:%ld, stream_num:%ld",
                        rt_ret, i, stream_num);
      GELOGE(RT_FAILED, "[Call][RtStreamCreateWithFlags] failed, ret:0x%X, stream_idx:%ld, stream_num:%ld",
             rt_ret, i, stream_num);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
    // Create slave stream, inactive by default, activated by hccl
    rt_ret = rtModelBindStream(davinci_model->GetRtModelHandle(), stream, RT_MODEL_WAIT_ACTIVE_STREAM);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtModelBindStream failed, ret:0x%X, stream_idx:%ld, stream_num:%ld",
                        rt_ret, i, stream_num);
      GELOGE(RT_FAILED, "[Call][RtModelBindStream] failed, ret:0x%X, stream_idx:%ld, stream_num:%ld",
             rt_ret, i, stream_num);
      (void)rtStreamDestroy(stream);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
    GELOGD("hccl_stream addr is=%p", stream);
    davinci_model->SaveHcclFollowStream(main_stream_id, stream);

    hccl_stream_list_.emplace_back(stream);
    davinci_model->PushHcclStream(stream);
  }
  GELOGI("CreateStream success.");
  return SUCCESS;
}

Status HcclTaskInfo::Distribute() {
  GELOGI("HcclTaskInfo Distribute Start. begin to call function LoadTask in hccl.");
  if (ops_kernel_store_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "Check param ops_kernel_store_ nullptr");
    GELOGE(INTERNAL_ERROR, "[Check][Param] ops kernel store is null.");
    return INTERNAL_ERROR;
  }
  OpsKernelInfoStore *ops_kernel_info_store = reinterpret_cast<OpsKernelInfoStore *>(ops_kernel_store_);
  GE_CHECK_NOTNULL(ops_kernel_info_store);
  GETaskInfo ge_task;
  TransToGETaskInfo(ge_task);
  auto result = ops_kernel_info_store->LoadTask(ge_task);
  if (result != HCCL_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Call ops_kernel_info_store LoadTask fail");
    GELOGE(INTERNAL_ERROR, "[Load][Task] fail, return ret:%u", result);
    return INTERNAL_ERROR;
  }
  GELOGI("HcclTaskInfo Distribute Success.");
  return SUCCESS;
}

Status HcclTaskInfo::CalculateArgs(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GE_CHECK_NOTNULL(davinci_model);
  auto hccl_def = task_def.kernel_hccl();
  uint32_t op_index = hccl_def.op_index();
  GELOGI("HcclTaskInfo Init, op_index is: %u", op_index);
  // Get HCCL op
  auto op_desc = davinci_model->GetOpByIndex(op_index);
  GE_CHECK_NOTNULL(op_desc);
  GELOGI("Calc opType[%s] args size. Node name is [%s]", op_desc->GetType().c_str(), op_desc->GetName().c_str());
  // Only need the number of addr to allocate args memory
  auto input_size = op_desc->GetInputsSize();
  auto output_size = op_desc->GetOutputsSize();
  auto workspace_size = op_desc->GetWorkspaceBytes().size();
  uint32_t args_size = sizeof(void *) * (input_size + output_size + workspace_size);
  args_offset_ = davinci_model->GetTotalArgsSize();
  davinci_model->SetTotalArgsSize(args_size);
  GELOGI("Calculate hccl task args , args_size %u, args_offset %u", args_size, args_offset_);
  return SUCCESS;
}

void HcclTaskInfo::SetIoAddrs(const OpDescPtr &op_desc) {
  const RuntimeParam &rts_param = davinci_model_->GetRuntimeParam();
  const auto input_data_addrs = ModelUtils::GetInputDataAddrs(rts_param, op_desc);
  const auto output_data_addrs = ModelUtils::GetOutputDataAddrs(rts_param, op_desc);
  const auto workspace_data_addrs = ModelUtils::GetWorkspaceDataAddrs(rts_param, op_desc);
  io_addrs_.insert(io_addrs_.end(), input_data_addrs.begin(), input_data_addrs.end());
  io_addrs_.insert(io_addrs_.end(), output_data_addrs.begin(), output_data_addrs.end());
  io_addrs_.insert(io_addrs_.end(), workspace_data_addrs.begin(), workspace_data_addrs.end());
}

Status HcclTaskInfo::UpdateArgs() {
  GELOGI("HcclTaskInfo::UpdateArgs in.");
  davinci_model_->SetTotalIOAddrs(io_addrs_);
  GELOGI("HcclTaskInfo::UpdateArgs success.");
  return SUCCESS;
}

Status HcclTaskInfo::SetAddrs(const std::shared_ptr<OpDesc> &op_desc,
                              std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  GE_CHK_STATUS_RET(HcomOmeUtil::CheckKernelHcclInfo(op_desc, kernel_hccl_infos),
                    "[Check][Param] HcomOmeUtil:: the number of GETaskKernelHcclInfo is invalid, node:%s(%s).",
                    op_desc->GetName().c_str(), op_desc->GetType().c_str());
  GELOGI("Set hccl task input output address, node[%s], type[%s] kernel_hccl_infos.size[%zu].",
         op_desc->GetName().c_str(), op_desc->GetType().c_str(), kernel_hccl_infos.size());
  if (op_desc->GetType() == HVDWAIT) {
    return SUCCESS;
  }

  HcclReduceOp op_type = HCCL_REDUCE_SUM;
  GE_CHECK_NOTNULL(davinci_model_);
  GELOGI("Calc opType[%s] input address before. Node name[%s]", op_desc->GetType().c_str(), op_desc->GetName().c_str());
  vector<void *> input_data_addrs;
  vector<void *> output_data_addrs;
  if (!davinci_model_->IsKnownNode()) {
    input_data_addrs = ModelUtils::GetInputDataAddrs(davinci_model_->GetRuntimeParam(), op_desc);
    output_data_addrs = ModelUtils::GetOutputDataAddrs(davinci_model_->GetRuntimeParam(), op_desc);
  }
  void *input_data_addr = nullptr;
  void *output_data_addr = nullptr;
  // initialize every kernel_hccl_info inputDataAddr
  for (size_t i = 0; i < kernel_hccl_infos.size(); i++) {
    std::string hccl_type = kernel_hccl_infos[i].hccl_type;
    if (davinci_model_->IsKnownNode()) {
      input_data_addr = reinterpret_cast<void *>(reinterpret_cast<uint64_t *>(args_) + i);
      output_data_addr = reinterpret_cast<void *>(reinterpret_cast<uint64_t *>(args_) + op_desc->GetInputsSize() + i);
      GELOGI("Hccl task info known input addr %p, output addr %p.", input_data_addr, output_data_addr);
    } else {
      input_data_addr = input_data_addrs.empty() ? nullptr : input_data_addrs[i];
      output_data_addr = output_data_addrs.empty() ? nullptr : output_data_addrs[i];
    }
    kernel_hccl_infos[i].inputDataAddr = input_data_addr;
    if (hccl_type == HCOMALLGATHER || hccl_type == HCOMRECEIVE || hccl_type == HVDCALLBACKALLGATHER) {
      kernel_hccl_infos[i].outputDataAddr = output_data_addr;
    } else if (hccl_type == HCOMALLREDUCE ||
               hccl_type == HCOMREDUCESCATTER || hccl_type == HVDCALLBACKALLREDUCE || hccl_type == HCOMREDUCE) {
      GE_CHK_STATUS_RET(HcomOmeUtil::GetHcclOperationType(op_desc, op_type),
                        "[Get][HcomOperationType] fail! op:%s", op_desc->GetName().c_str());
      kernel_hccl_infos[i].outputDataAddr = output_data_addr;
      kernel_hccl_infos[i].opType = op_type;
    }
    davinci_model_->DisableZeroCopy(input_data_addr);
  }
  return SUCCESS;
}
void HcclTaskInfo::TransToGETaskInfo(GETaskInfo &ge_task) {
  ge_task.id = id_;
  ge_task.type = static_cast<uint16_t>(RT_MODEL_TASK_HCCL);
  ge_task.stream = stream_;
  ge_task.kernelHcclInfo = kernel_hccl_infos_;
  ge_task.privateDef = private_def_;
  ge_task.privateDefLen = private_def_len_;
  ge_task.opsKernelStorePtr = ops_kernel_store_;
  for (size_t i = 0; i < ge_task.kernelHcclInfo.size(); i++) {
    ge_task.kernelHcclInfo[i].hcclStreamList = hccl_stream_list_;
  }
}
void HcclTaskInfo::GetPrivateDefByTaskDef(const domi::TaskDef &task) {
  // Get privateDef and opsKernelStorePtr from taskDef and save them in taskInfo
  GELOGI("get custom info in modelTaskDef.");
  ops_kernel_store_ = nullptr;
  void *ops_kernel_store_name_temp = reinterpret_cast<void *>(static_cast<uintptr_t>(task.ops_kernel_store_ptr()));
  if (ops_kernel_store_name_temp != nullptr) {
    ops_kernel_store_ = std::move(ops_kernel_store_name_temp);
    std::string private_def_temp = task.private_def();
    if (!private_def_temp.empty()) {
      private_def_len_ = private_def_temp.size();
      rtError_t ret = rtMallocHost(&private_def_, private_def_len_);
      if (ret != RT_ERROR_NONE) {
        REPORT_CALL_ERROR("E19999", "Call rtMallocHost failed, ret:0x%X, size:%u", ret, private_def_len_);
        GELOGE(RT_FAILED, "[Call][RtMallocHost] Fail, ret:0x%X, size:%u", ret, private_def_len_);
        return;
      }

      ret = rtMemcpy(private_def_, private_def_len_, task.private_def().c_str(), private_def_len_,
                     RT_MEMCPY_HOST_TO_HOST);
      if (ret != RT_ERROR_NONE) {
        REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, ret:0x%X, size:%u", ret, private_def_len_);
        GELOGE(RT_FAILED, "[Call][RtMemcpy] Fail, ret:0x%X, size:%u", ret, private_def_len_);
        return;
      }
      GELOGI("The first address of the custom info, privateDef=%p.", private_def_);
    }
  }
}
void HcclTaskInfo::CreateKernelHcclInfo(const ge::ConstOpDescPtr &op_desc) {
  GE_CHECK_NOTNULL_JUST_RETURN(op_desc);
  if (HcomOmeUtil::IsHCOMOp(op_desc->GetType())) {
    GETaskKernelHcclInfo kernel_hccl_info;
    kernel_hccl_infos_.emplace_back(kernel_hccl_info);
  } else if (HcomOmeUtil::IsHorovodOp(op_desc->GetType())) {
    // Horovod wait do not have any input, but create a GETaskKernelHcclInfo to record hccl_type.
    // Other Operator need to check that the number of GETaskKernelHcclInfo must equals to number of inputs
    if (op_desc->GetType() == HVDWAIT) {
      GETaskKernelHcclInfo kernel_hccl_info;
      kernel_hccl_infos_.emplace_back(kernel_hccl_info);
      return;
    }
    for (size_t i = 0; i < op_desc->GetInputsSize(); i++) {
      GETaskKernelHcclInfo kernel_hccl_info;
      kernel_hccl_infos_.emplace_back(kernel_hccl_info);
    }
  }
}
Status HcclTaskInfo::SetWorkspace(const std::shared_ptr<OpDesc> &op_desc,
                                  std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  GE_CHECK_NOTNULL(davinci_model_);
  GELOGI("SetWorkspace Node[%s] opType[%s] set workspace.", op_desc->GetName().c_str(), op_desc->GetType().c_str());
  uint64_t workspace_mem_size = 0;
  void *workspace_addr = nullptr;
  auto workspace_bytes = op_desc->GetWorkspaceBytes();
  if (!workspace_bytes.empty()) {
    uint64_t workspace_mem_size_tmp = workspace_bytes[0];
    GELOGI("hccl need workSpaceMemSize=%lu", workspace_mem_size_tmp);
    if (workspace_mem_size_tmp != 0) {
      workspace_mem_size = workspace_mem_size_tmp;
      if (davinci_model_->IsKnownNode()) {
        workspace_addr = reinterpret_cast<void *>(reinterpret_cast<uint64_t *>(args_) + op_desc->GetInputsSize() +
                                                  op_desc->GetOutputsSize());
      } else {
        const auto workspace_data_addrs = ModelUtils::GetWorkspaceDataAddrs(davinci_model_->GetRuntimeParam(), op_desc);
        workspace_addr = workspace_data_addrs.empty() ? nullptr : workspace_data_addrs[0];
      }
    }
  }
  for (size_t i = 0; i < kernel_hccl_infos.size(); i++) {
    kernel_hccl_infos[i].workSpaceMemSize = workspace_mem_size;
    kernel_hccl_infos[i].workSpaceAddr = workspace_addr;
  }
  return SUCCESS;
}
REGISTER_TASK_INFO(RT_MODEL_TASK_HCCL, HcclTaskInfo);
}  // namespace ge
