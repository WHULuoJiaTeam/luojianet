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

#include "single_op/task/aicpu_task_builder.h"
#include <vector>
#include "single_op/task/build_task_utils.h"
#include "runtime/mem.h"
#include "framework/common/debug/ge_log.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/load/model_manager/model_manager.h"

namespace ge {
  AiCpuTaskBuilder::AiCpuTaskBuilder(const OpDescPtr &op_desc, const domi::KernelExDef &kernel_def)
      : op_desc_(op_desc), kernel_def_(kernel_def) {}

  Status AiCpuTaskBuilder::SetFmkOpKernel(void *io_addr, void *ws_addr, STR_FWK_OP_KERNEL &fwk_op_kernel) {
    auto sec_ret = memcpy_s(&fwk_op_kernel, sizeof(STR_FWK_OP_KERNEL),
                            kernel_def_.args().data(), kernel_def_.args().size());
    if (sec_ret != EOK) {
      GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Memcpy_s][Param:fwk_op_kernel] failed, ret: %d", sec_ret);
      REPORT_INNER_ERROR("E19999", "memcpy_s fwk_op_kernel failed, ret:%d.", sec_ret);
      return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
    }

    auto io_addr_val = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(io_addr));
    fwk_op_kernel.fwkKernelBase.fwk_kernel.inputOutputAddr = io_addr_val;
    auto ws_addr_val = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ws_addr));
    fwk_op_kernel.fwkKernelBase.fwk_kernel.workspaceBaseAddr = ws_addr_val;
    return SUCCESS;
  }

  Status AiCpuTaskBuilder::SetKernelArgs(void **args, STR_FWK_OP_KERNEL &fwk_op_kernel) {
    void *fwk_op_args = nullptr;
    auto rt_ret = rtMalloc(&fwk_op_args, sizeof(STR_FWK_OP_KERNEL), RT_MEMORY_HBM);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(rt_ret, "[RtMalloc][Memory] failed, ret = %d", rt_ret);
      REPORT_INNER_ERROR("E19999", "rtMalloc Memory failed, ret = %d", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }

    rt_ret = rtMemcpy(fwk_op_args, sizeof(STR_FWK_OP_KERNEL), &fwk_op_kernel,
                      sizeof(STR_FWK_OP_KERNEL), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      (void)rtFree(fwk_op_args);
      GELOGE(rt_ret, "[rtMemcpy][Fwk_Op_Args] failed, ret = %d", rt_ret);
      REPORT_INNER_ERROR("E19999", "rtMemcpy fwk_op_args failed, ret = %d", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
    *args = fwk_op_args;
    return SUCCESS;
  }

  Status AiCpuTaskBuilder::InitWorkspaceAndIO(AiCpuTask &task, const SingleOpModelParam &param) {
    if (kernel_def_.args_size() > sizeof(STR_FWK_OP_KERNEL)) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Size]sizeof STR_FWK_OP_KERNEL is: %lu, but args_size is: %d",
          sizeof(STR_FWK_OP_KERNEL), kernel_def_.args_size());
      REPORT_INNER_ERROR("E19999", "sizeof STR_FWK_OP_KERNEL is: %lu, but args_size is: %d",
          sizeof(STR_FWK_OP_KERNEL), kernel_def_.args_size());
      return ACL_ERROR_GE_PARAM_INVALID;
    }
    GE_CHK_RT_RET(rtMalloc(&task.workspace_addr_, kernel_def_.task_info_size(), RT_MEMORY_HBM));
    GE_CHK_RT_RET(rtMemcpy(task.workspace_addr_, kernel_def_.task_info_size(),
                           kernel_def_.task_info().data(), kernel_def_.task_info_size(),
                           RT_MEMCPY_HOST_TO_DEVICE));

    auto addresses = BuildTaskUtils::GetAddresses(op_desc_, param, false);
    task.io_addr_host_ = BuildTaskUtils::JoinAddresses(addresses);
    task.io_addr_size_ = task.io_addr_host_.size() * sizeof(void *);
    GE_CHK_RT_RET(rtMalloc(&task.io_addr_, task.io_addr_size_, RT_MEMORY_HBM));
    return SUCCESS;
  }

  Status AiCpuTaskBuilder::BuildTask(ge::AiCpuTask &task, const SingleOpModelParam &param, uint64_t kernel_id) {
    GE_CHK_STATUS_RET_NOLOG(InitWorkspaceAndIO(task, param));

    STR_FWK_OP_KERNEL fwk_op_kernel = {0};
    auto ret = SetFmkOpKernel(task.io_addr_, task.workspace_addr_, fwk_op_kernel);
    if (ret != SUCCESS) {
      return ret;
    }

    GE_CHECK_NOTNULL(op_desc_);
    task.op_desc_ = op_desc_;
    task.num_inputs_ = op_desc_->GetInputsSize();
    task.num_outputs_ = op_desc_->GetOutputsSize();

    // get kernel_ext_info
    auto &kernel_ext_info = kernel_def_.kernel_ext_info();
    auto kernel_ext_info_size = kernel_def_.kernel_ext_info_size();
    GE_CHK_BOOL_RET_STATUS(kernel_ext_info.size() == kernel_ext_info_size, ACL_ERROR_GE_PARAM_INVALID,
                           "[Check][Size]task def kernel_ext_info.size=%zu, but kernel_ext_info_size=%u.",
                           kernel_ext_info.size(), kernel_ext_info_size);
    GE_CHK_STATUS_RET(task.SetExtInfoAndType(kernel_ext_info, kernel_id), "[Set][ExtInfoAndType]failed.");

    if (task.ext_info_addr_dev_ != nullptr) {
      fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoAddr =  reinterpret_cast<uintptr_t>(task.ext_info_addr_dev_);
      fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoLen = kernel_ext_info_size;
    }
    GE_CHK_STATUS_RET(task.SetInputConst(), "[Set][InputConst] failed.");
    GE_CHK_STATUS_RET(task.InitForSummaryAndCopy(), "[Init][SummaryAndCopy] failed.");

    fwk_op_kernel.fwkKernelBase.fwk_kernel.sessionID = ULLONG_MAX;
    fwk_op_kernel.fwkKernelBase.fwk_kernel.kernelID = kernel_id;
    fwk_op_kernel.fwkKernelBase.fwk_kernel.opType = aicpu::FWKAdapter::FWKOperateType::FWK_ADPT_KERNEL_RUN_NO_SESS;
    ret = SetKernelArgs(&task.args_, fwk_op_kernel);
    if (ret != SUCCESS) {
      return ret;
    }

    task.arg_size_ = sizeof(STR_FWK_OP_KERNEL);
    task.op_type_ = op_desc_->GetName();
    task.task_info_ = kernel_def_.task_info();
    task.kernel_id_ = kernel_id;

    auto debug_info = BuildTaskUtils::GetTaskInfo(op_desc_);
    GELOGI("[TASK_INFO] %lu/%s %s", kernel_id, task.op_type_.c_str(), debug_info.c_str());
    return SUCCESS;
  }
}  // namespace ge
