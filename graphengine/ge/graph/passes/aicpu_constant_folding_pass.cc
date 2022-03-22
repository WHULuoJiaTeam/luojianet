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

#include "graph/passes/aicpu_constant_folding_pass.h"

#include <memory>
#include <vector>

#include "framework/common/debug/log.h"
#include "common/ge/ge_util.h"
#include "framework/common/types.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "init/gelib.h"
#include "opskernel_manager/ops_kernel_builder_manager.h"

namespace {
const char *const kKernelLibName = "aicpu_tf_kernel";
const char *const kNotSupported = "0";
const uint64_t kReleaseFlag = 1;
const uint64_t kOpsFlag = 1;
const uint64_t kDouble = 2;
}  // namespace
namespace ge {
Status AicpuConstantFoldingPass::Run(ge::NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  GELOGD("Start aicpu constant folding on node [%s]", node->GetName().c_str());
  if (IsSkipFold(node)) {
    return SUCCESS;
  }
  vector<ConstGeTensorPtr> weight_vec;
  bool flag = CheckInput(node, weight_vec);
  if (!flag) {
    return SUCCESS;
  }
  OpDescPtr node_desc = node->GetOpDesc();  // checked before
  vector<DataPtrInfo> data_vec;
  vector<AddrAndType> input_addrs;
  vector<uint64_t> output_addrs;
  Status ret = GetInputAddrs(weight_vec, input_addrs);
  if (ret != SUCCESS) {
    ReleaseMemory(input_addrs, output_addrs, data_vec);
    return SUCCESS;
  }

  ret = GetOutputAddrs(node_desc, output_addrs);
  if (ret != SUCCESS) {
    ReleaseMemory(input_addrs, output_addrs, data_vec);
    return SUCCESS;
  }

  ret = LaunchSingleOpRunTask(node, input_addrs, output_addrs);
  if (ret != SUCCESS) {
    ReleaseMemory(input_addrs, output_addrs, data_vec);
    return SUCCESS;
  }
  GELOGI("[Node:%s] Launch singleOpRunTask success", node->GetName().c_str());

  vector<uint64_t> data_infos;
  ret = GenerateDataPtrInfo(output_addrs, data_vec, data_infos);
  if (ret != SUCCESS) {
    ReleaseMemory(input_addrs, output_addrs, data_vec);
    return SUCCESS;
  }
  GELOGI("[Node:%s] Generate dataPtrInfo success", node->GetName().c_str());

  ret = LaunchMemCopyTask(data_infos);
  if (ret != SUCCESS) {
    ReleaseMemory(input_addrs, output_addrs, data_vec);
    return SUCCESS;
  }
  GELOGI("[Node:%s] Launch memCopyTask success", node->GetName().c_str());

  vector<GeTensorPtr> outputs;
  ret = GenerateGeTensor(node_desc, data_vec, outputs);
  if (ret != SUCCESS) {
    ReleaseMemory(input_addrs, output_addrs, data_vec);
    return SUCCESS;
  }
  ReleaseMemory(input_addrs, output_addrs, data_vec);
  GELOGI("[Node:%s] Generate geTensor success", node->GetName().c_str());
  return Folding(node, outputs);
}

bool AicpuConstantFoldingPass::CheckInput(const NodePtr &node, vector<ConstGeTensorPtr> &weight_vec) {
  OpDescPtr node_desc = node->GetOpDesc();
  if (node_desc == nullptr) {
    GELOGW("Opdesc of %s is null", node->GetName().c_str());
    return false;
  }
  DataType data_type = node_desc->GetOutputDesc(0).GetDataType();
  Format format = node_desc->GetOutputDesc(0).GetFormat();
  GELOGD("Current [node:%s, type:%s] info: format: %s, datatype:%s", node->GetName().c_str(), node->GetType().c_str(),
         TypeUtils::FormatToSerialString(format).c_str(), TypeUtils::DataTypeToSerialString(data_type).c_str());
  auto input_nodes = OpDescUtils::GetConstInputNode(*node);
  if (input_nodes.empty() || input_nodes.size() != node_desc->GetInputsSize()) {
    GELOGD("Const input nodes size is %zu, and nodeDesc inputsSize is %zu, skip fold.", input_nodes.size(),
           node_desc->GetInputsSize());
    return false;
  }
  weight_vec = OpDescUtils::GetInputData(input_nodes);
  return true;
}

Status AicpuConstantFoldingPass::GetInputAddrs(const vector<ConstGeTensorPtr> &weight_vec,
                                               vector<AddrAndType> &input_addrs) {
  if (weight_vec.empty()) {
    REPORT_INNER_ERROR("E19999", "Param weight_vec is empty, check invalid");
    GELOGE(FAILED, "[Check][Param] Weight is null");
    return FAILED;
  }
  for (const ConstGeTensorPtr &weight : weight_vec) {
    void *input_addr = nullptr;
    GE_CHK_RT_RET(rtMalloc(&input_addr, weight->GetData().size(), RT_MEMORY_HBM));

    rtError_t rt_ret = rtMemcpy(input_addr, weight->GetData().size(), weight->GetData().data(),
                                weight->GetData().size(), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%zu, ret = 0x%X",
                        weight->GetData().size(), rt_ret);
      GELOGE(rt_ret, "[Call][RtMemcpy] failed, size:%zu, ret = 0x%X", weight->GetData().size(), rt_ret);
      GE_CHK_RT(rtFree(input_addr));
      return FAILED;
    }

    AddrAndType input_info = {static_cast<uint64_t>(reinterpret_cast<uintptr_t>(input_addr)), kData};
    input_addrs.emplace_back(input_info);
  }
  return SUCCESS;
}

Status AicpuConstantFoldingPass::GetOutputAddrs(const OpDescPtr &node_desc, vector<uint64_t> &output_addrs) {
  if (node_desc->GetOutputsSize() == 0) {
    REPORT_INNER_ERROR("E19999", "Ouput desc size of op:%s(%s) is 0, check invalid",
                       node_desc->GetName().c_str(), node_desc->GetType().c_str());
    GELOGE(FAILED, "[Get][OutputsSize] Ouput desc size of op:%s(%s) is 0",
           node_desc->GetName().c_str(), node_desc->GetType().c_str());
    return FAILED;
  }
  for (size_t i = 0; i < node_desc->GetOutputsSize(); ++i) {
    void *summary_addr = nullptr;
    GE_CHK_RT_RET(rtMalloc(&summary_addr, sizeof(aicpu::FWKAdapter::ResultSummary), RT_MEMORY_HBM));
    output_addrs.emplace_back(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(summary_addr)));
  }
  return SUCCESS;
}

Status AicpuConstantFoldingPass::GenerateDataPtrInfo(const vector<uint64_t> &output_addrs,
                                                     vector<DataPtrInfo> &data_vec, vector<uint64_t> &data_infos) {
  for (uint64_t output_addr : output_addrs) {
    aicpu::FWKAdapter::ResultSummary result_summary;
    GE_CHK_RT_RET(rtMemcpy(&result_summary, sizeof(aicpu::FWKAdapter::ResultSummary),
                           reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(output_addr)),
                           sizeof(aicpu::FWKAdapter::ResultSummary), RT_MEMCPY_DEVICE_TO_HOST));
    void *raw_data_addr = nullptr;
    GE_CHK_RT_RET(rtMalloc(&raw_data_addr, result_summary.raw_data_size, RT_MEMORY_HBM));

    void *shape_data_addr = nullptr;
    // shape_data_size = 0 means scalar
    if (result_summary.shape_data_size != 0) {
      rtError_t rt_ret = rtMalloc(&shape_data_addr, result_summary.shape_data_size, RT_MEMORY_HBM);
      if (rt_ret != RT_ERROR_NONE) {
        REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%lu, ret = 0x%X",
                          result_summary.shape_data_size, rt_ret);
        GELOGE(rt_ret, "[Call][RtMalloc] failed, size:%lu, ret = 0x%X", result_summary.shape_data_size, rt_ret);
        GE_CHK_RT(rtFree(raw_data_addr));
        return FAILED;
      }
    }

    DataPtrInfo raw_data_info;
    raw_data_info.release_flag = kReleaseFlag;
    raw_data_info.data_size = result_summary.raw_data_size;
    raw_data_info.src_ptr = result_summary.raw_data_ptr;
    raw_data_info.dst_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(raw_data_addr));
    data_vec.emplace_back(raw_data_info);

    DataPtrInfo shape_data_info;
    shape_data_info.release_flag = kReleaseFlag;
    shape_data_info.data_size = result_summary.shape_data_size;
    shape_data_info.src_ptr = result_summary.shape_data_ptr;
    shape_data_info.dst_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(shape_data_addr));
    data_vec.emplace_back(shape_data_info);
  }
  for (const DataPtrInfo &data_info : data_vec) {
    data_infos.emplace_back(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&data_info)));
  }
  return SUCCESS;
}

Status AicpuConstantFoldingPass::UpdateWorkSpaceAddr(string &task_info, STR_FWK_OP_KERNEL &task) {
  // Update the workspace_addr
  if (task_info.empty()) {
    REPORT_INNER_ERROR("E19999", "Param task_info is empty, check invalid");
    GELOGE(FAILED, "[Check][Param] task_info is empty ");
    return FAILED;
  }
  void *workspace_addr = nullptr;
  GE_CHK_RT_RET(rtMalloc(&workspace_addr, task_info.size(), RT_MEMORY_HBM));
  rtError_t rt_ret =
    rtMemcpy(workspace_addr, task_info.size(), task_info.data(), task_info.size(), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%zu, ret = 0x%X",
                      task_info.size(), rt_ret);
    GELOGE(rt_ret, "[Call][RtMemcpy] failed, size:%zu, ret = 0x%X", task_info.size(), rt_ret);
    GE_CHK_RT(rtFree(workspace_addr));
    return FAILED;
  }

  uint64_t workspace_base_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(workspace_addr));
  task.fwkKernelBase.fwk_kernel.workspaceBaseAddr = workspace_base_addr;
  return SUCCESS;
}

Status AicpuConstantFoldingPass::UpdateInputAndOutputAddr(const vector<uint64_t> &io_addrs, STR_FWK_OP_KERNEL &task) {
  auto addrs_size = sizeof(uint64_t) * (io_addrs.size());
  if (addrs_size <= 0) {
    REPORT_INNER_ERROR("E19999", "Param io_addrs size is 0, check invalid");
    GELOGE(FAILED, "[Check][Param] addrs_size is less than 1 ");
    return FAILED;
  }
  void *input_output_addr = nullptr;
  GE_CHK_RT_RET(rtMalloc(&input_output_addr, addrs_size, RT_MEMORY_HBM));
  rtError_t rt_ret = rtMemcpy(input_output_addr, addrs_size, io_addrs.data(), addrs_size, RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%zu, ret = 0x%X",
                      addrs_size, rt_ret);
    GELOGE(rt_ret, "[Call][RtMemcpy] failed, size:%zu, ret = 0x%X", addrs_size, rt_ret);
    GE_CHK_RT(rtFree(input_output_addr));
    return FAILED;
  }

  uint64_t in_out_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(input_output_addr));
  task.fwkKernelBase.fwk_kernel.inputOutputAddr = in_out_addr;
  return SUCCESS;
}

Status AicpuConstantFoldingPass::UpdateSingleOpAddr(string &task_info, const vector<AddrAndType> &input_addrs,
                                                    const vector<uint64_t> &outputs_addr_vec, STR_FWK_OP_KERNEL &task) {
  // Build the SingleOpAddr
  vector<uint64_t> inputs_addr_vec;
  for (const auto &item : input_addrs) {
    inputs_addr_vec.push_back(item.input_addr);
  }
  vector<uint64_t> io_addrs;
  io_addrs.insert(io_addrs.end(), inputs_addr_vec.begin(), inputs_addr_vec.end());
  io_addrs.insert(io_addrs.end(), outputs_addr_vec.begin(), outputs_addr_vec.end());

  Status ret = UpdateInputAndOutputAddr(io_addrs, task);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Update][InputAndOutputAddr] failed, ret:%d", ret);
    return ret;
  }
  ret = UpdateWorkSpaceAddr(task_info, task);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Update][WorkSpaceAddr] failed, ret:%d", ret);
    return ret;
  }
  return SUCCESS;
}

Status AicpuConstantFoldingPass::UpdateMemCopyAddr(string &task_info, const vector<uint64_t> &data_infos,
                                                   vector<uint64_t> &internal_addrs, STR_FWK_OP_KERNEL &task) {
  vector<uint64_t> release_flags;
  vector<uint64_t> data_sizes;
  vector<uint64_t> src_addrs;
  vector<uint64_t> dst_addrs;
  for (auto item : data_infos) {
    auto *data_info_ptr = reinterpret_cast<DataPtrInfo *>(reinterpret_cast<uintptr_t>(item));  // pointer cannot be null
    release_flags.push_back(data_info_ptr->release_flag);
    data_sizes.push_back(data_info_ptr->data_size);
    src_addrs.push_back(data_info_ptr->src_ptr);
    dst_addrs.push_back(data_info_ptr->dst_ptr);
  }
  vector<vector<uint64_t>> inputs = {release_flags, data_sizes, src_addrs, dst_addrs};
  auto data_size = sizeof(uint64_t) * (data_infos.size());
  vector<uint64_t> io_addrs;
  if (data_infos.size() > 0) {
    for (const auto &item : inputs) {
      void *input_addr_ptr = nullptr;
      GE_CHK_RT_RET(rtMalloc(&input_addr_ptr, data_size, RT_MEMORY_HBM));
      rtError_t rt_ret = rtMemcpy(input_addr_ptr, data_size, item.data(), data_size, RT_MEMCPY_HOST_TO_DEVICE);
      if (rt_ret != RT_ERROR_NONE) {
        REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%zu, ret = 0x%X",
                          data_size, rt_ret);
        GELOGE(rt_ret, "[Call][RtMemcpy] failed, size:%zu, ret = 0x%X", data_size, rt_ret);
        GE_CHK_RT(rtFree(input_addr_ptr));
        return FAILED;
      }
      uint64_t input_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(input_addr_ptr));
      io_addrs.push_back(input_addr);
    }
  }
  internal_addrs = io_addrs;

  Status ret = UpdateInputAndOutputAddr(io_addrs, task);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Update][InputAndOutputAddr] failed, ret:%d", ret);
    return ret;
  }
  ret = UpdateWorkSpaceAddr(task_info, task);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Update][WorkSpaceAddr] failed, ret:%d", ret);
    return ret;
  }
  return SUCCESS;
}

Status AicpuConstantFoldingPass::LaunchSingleOpRunTask(const NodePtr &node, const vector<AddrAndType> &input_addrs,
                                                       const vector<uint64_t> &output_addrs) {
  void *task_buf = nullptr;
  auto instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    REPORT_INNER_ERROR("E19999", "GeLib is not init before, check invalid");
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Check][Param] GE is not initialized");
    return GE_CLI_GE_NOT_INITIALIZED;
  }
  auto kernel_builder = OpsKernelBuilderManager::Instance().GetOpsKernelBuilder(kKernelLibName);
  if (kernel_builder == nullptr) {
    REPORT_INNER_ERROR("E19999", "Find ops kernel by name:%s failed",
                       kKernelLibName);
    GELOGE(FAILED, "[Get][OpsKernelBuilder] by name:%s failed", kKernelLibName);
    return FAILED;
  }
  STR_FWK_OP_KERNEL aicpu_task;
  aicpu_task.fwkKernelBase.fwk_kernel.inputOutputAddr = 0;
  aicpu_task.fwkKernelBase.fwk_kernel.workspaceBaseAddr = 0;
  aicpu_task.fwkKernelBase.fwk_kernel.extInfoAddr = 0;
  aicpu_task.fwkKernelBase.fwk_kernel.extInfoLen = 0;
  std::string task_info;
  Status ret = kernel_builder->GenSingleOpRunTask(node, aicpu_task, task_info);
  if (ret != SUCCESS) {
    return ret;
  }
  std::function<void()> callback = [&]() {
    void *input_output_ptr =
      reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(aicpu_task.fwkKernelBase.fwk_kernel.inputOutputAddr));
    if (input_output_ptr != nullptr) {
      GE_CHK_RT(rtFree(input_output_ptr));
    }
    void *workspace_addr_ptr =
      reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(aicpu_task.fwkKernelBase.fwk_kernel.workspaceBaseAddr));
    if (workspace_addr_ptr != nullptr) {
      GE_CHK_RT(rtFree(workspace_addr_ptr));
    }
  };
  GE_MAKE_GUARD(release, callback);

  ret = UpdateSingleOpAddr(task_info, input_addrs, output_addrs, aicpu_task);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Update][SingleOpAddr] failed, ret:%d", ret);
    return ret;
  }
  ret = GenerateTaskForLaunch(aicpu_task, task_buf);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Generate][Task] For Launch failed, ret:%d", ret);
    return ret;
  }
  ret = KernelLaunch(task_buf);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][KernelLaunch] failed, ret:%d", ret);
    return ret;
  }

  return SUCCESS;
}

Status AicpuConstantFoldingPass::LaunchMemCopyTask(const vector<uint64_t> &data_infos) {
  void *task_buf = nullptr;
  auto instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    REPORT_INNER_ERROR("E19999", "GeLib is not init before, check invalid");
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Check][Param] GE is not initialized");
    return GE_CLI_GE_NOT_INITIALIZED;
  }
  auto kernel_builder = OpsKernelBuilderManager::Instance().GetOpsKernelBuilder(kKernelLibName);
  if (kernel_builder == nullptr) {
    REPORT_INNER_ERROR("E19999", "Find ops kernel by name:%s failed",
                       kKernelLibName);
    GELOGE(FAILED, "[Get][OpsKernelBuilder] by name:%s failed", kKernelLibName);
    return FAILED;
  }
  STR_FWK_OP_KERNEL aicpu_task;
  aicpu_task.fwkKernelBase.fwk_kernel.inputOutputAddr = 0;
  aicpu_task.fwkKernelBase.fwk_kernel.workspaceBaseAddr = 0;
  aicpu_task.fwkKernelBase.fwk_kernel.extInfoAddr = 0;
  aicpu_task.fwkKernelBase.fwk_kernel.extInfoLen = 0;
  std::string task_info;
  Status ret = kernel_builder->GenMemCopyTask(data_infos.size(), aicpu_task, task_info);
  if (ret != SUCCESS) {
    return ret;
  }

  vector<uint64_t> internal_addrs;
  std::function<void()> callback = [&]() {
    for (auto item : internal_addrs) {
      GE_CHK_RT(rtFree(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(item))));  // pointer cannot be null
    }
    void *input_output_ptr =
      reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(aicpu_task.fwkKernelBase.fwk_kernel.inputOutputAddr));
    if (input_output_ptr != nullptr) {
      GE_CHK_RT(rtFree(input_output_ptr));
    }
    void *workspace_addr_ptr =
      reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(aicpu_task.fwkKernelBase.fwk_kernel.workspaceBaseAddr));
    if (workspace_addr_ptr != nullptr) {
      GE_CHK_RT(rtFree(workspace_addr_ptr));
    }
  };
  GE_MAKE_GUARD(release, callback);

  ret = UpdateMemCopyAddr(task_info, data_infos, internal_addrs, aicpu_task);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Update][MemCopyAddr] failed, ret:%d", ret);
    return ret;
  }
  ret = GenerateTaskForLaunch(aicpu_task, task_buf);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Generate][Task] For Launch failed, ret:%d", ret);
    return ret;
  }
  ret = KernelLaunch(task_buf);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][KernelLaunch] failed, ret:%d", ret);
    return ret;
  }
  return SUCCESS;
}

Status AicpuConstantFoldingPass::GenerateTaskForLaunch(STR_FWK_OP_KERNEL &aicpu_task, void *&task_buf) {
  GE_CHK_RT_RET(rtMalloc(&task_buf, sizeof(STR_FWK_OP_KERNEL), RT_MEMORY_HBM));

  rtError_t rt_ret = rtMemcpy(task_buf, sizeof(STR_FWK_OP_KERNEL), reinterpret_cast<void *>(&aicpu_task),
                              sizeof(STR_FWK_OP_KERNEL), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%zu, ret = 0x%X",
                      sizeof(STR_FWK_OP_KERNEL), rt_ret);
    GELOGE(rt_ret, "[Call][RtMemcpy] failed, size:%zu, ret = 0x%X", sizeof(STR_FWK_OP_KERNEL), rt_ret);
    GE_CHK_RT(rtFree(task_buf));
    return FAILED;
  }
  return SUCCESS;
}

Status AicpuConstantFoldingPass::KernelLaunch(void *task_buf) {
  rtModel_t model = nullptr;
  rtStream_t stream = nullptr;
  rtStream_t stream_run = nullptr;
  std::function<void()> callback = [&]() {
    if (task_buf != nullptr) {
      GE_CHK_RT(rtFree(task_buf));
    }
    if (model != nullptr) {
      GE_CHK_RT(rtModelDestroy(model));
    }
    if (stream != nullptr) {
      GE_CHK_RT(rtStreamDestroy(stream));
    }
    if (stream_run != nullptr) {
      GE_CHK_RT(rtStreamDestroy(stream_run));
    }
  };
  GE_MAKE_GUARD(release, callback);

  rtError_t rt_ret = rtModelCreate(&model, 0);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtModelCreate failed, ret = 0x%X",
                      rt_ret);
    GELOGE(rt_ret, "[Create][Model] failed, ret = 0x%X", rt_ret);
    return FAILED;
  }
  rt_ret = rtStreamCreate(&stream, 0);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtStreamCreate failed, ret = 0x%X",
                      rt_ret);
    GELOGE(rt_ret, "[Create][Stream] failed, ret = 0x%X", rt_ret);
    return FAILED;
  }
  rt_ret = rtModelBindStream(model, stream, 0);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtModelBindStream failed, ret = 0x%X",
                      rt_ret);
    GELOGE(rt_ret, "[Call][RtModelBindStream] failed, ret = 0x%X", rt_ret);
    return FAILED;
  }
  rt_ret = rtKernelLaunchEx(task_buf, sizeof(STR_FWK_OP_KERNEL), 0, stream);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtModelBindStream failed, ret = 0x%X",
                      rt_ret);
    GELOGE(rt_ret, "[Call][RtModelBindStream] failed, ret = 0x%X", rt_ret);
    return FAILED;
  }
  rt_ret = rtModelLoadComplete(model);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtModelLoadComplete failed, ret = 0x%X",
                      rt_ret);
    GELOGE(rt_ret, "[Call][RtModelLoadComplete] failed, ret = 0x%X", rt_ret);
    return FAILED;
  }
  rt_ret = rtStreamCreate(&stream_run, 0);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtStreamCreate failed, ret = 0x%X",
                      rt_ret);
    GELOGE(rt_ret, "[Call][RtStreamCreate] failed, ret = 0x%X", rt_ret);
    return FAILED;
  }
  rt_ret = rtModelExecute(model, stream_run, 0);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtModelExecute failed, ret = 0x%X",
                      rt_ret);
    GELOGE(rt_ret, "[Call][RtModelExecute] failed, ret = 0x%X", rt_ret);
    return FAILED;
  }
  rt_ret = rtStreamSynchronize(stream_run);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtStreamSynchronize failed, ret = 0x%X",
                      rt_ret);
    GELOGE(rt_ret, "[Call][RtStreamSynchronize] failed, ret = 0x%X", rt_ret);
    return FAILED;
  }
  return SUCCESS;
}

Status AicpuConstantFoldingPass::GenerateGeTensor(const OpDescPtr &node_desc, const vector<DataPtrInfo> &data_vec,
                                                  vector<GeTensorPtr> &outputs) {
  if ((node_desc->GetOutputsSize() * kDouble) != data_vec.size()) {
    REPORT_INNER_ERROR("E19999", "Output desc size:%zu of op:%s(%s), after multi 2, not equal to data_vec.size:%zu, "
                       "check invalid", node_desc->GetOutputsSize(),
                       node_desc->GetName().c_str(), node_desc->GetType().c_str(), data_vec.size());
    GELOGE(FAILED, "[Check][Param] Output desc size:%zu of op:%s(%s), after multi 2, not equal to data_vec.size:%zu",
           node_desc->GetOutputsSize(), node_desc->GetName().c_str(), node_desc->GetType().c_str(), data_vec.size());
    return FAILED;
  }

  for (size_t i = 0; i < node_desc->GetOutputsSize(); i++) {
    auto output_tensor_desc = node_desc->GetOutputDesc(static_cast<uint32_t>(i));
    GeTensorPtr output_ptr = MakeShared<GeTensor>(output_tensor_desc);
    if (output_ptr == nullptr) {
      REPORT_CALL_ERROR("E19999", "New GeTensor failed");
      GELOGE(FAILED, "[New][GeTensor] failed");
      return FAILED;
    }
    const DataPtrInfo &raw_data_info = data_vec.at(i * kDouble);
    uint64_t raw_data_size = raw_data_info.data_size;
    std::unique_ptr<uint8_t[]> data_addr(new (std::nothrow) uint8_t[raw_data_size]());
    if (data_addr == nullptr) {
      REPORT_CALL_ERROR("E19999", "New Buffer failed, size:%lu",
                        raw_data_size);
      GELOGE(MEMALLOC_FAILED, "[New][Buffer] failed, size:%lu", raw_data_size);
      return INTERNAL_ERROR;
    }
    GE_CHK_RT_RET(rtMemcpy(data_addr.get(), raw_data_size,
                           reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(raw_data_info.dst_ptr)), raw_data_size,
                           RT_MEMCPY_DEVICE_TO_HOST));
    GE_IF_BOOL_EXEC(output_ptr->SetData(data_addr.get(), raw_data_size) != GRAPH_SUCCESS,
                    GELOGE(FAILED, "[Set][Data] for node:%s output[%zu] failed", node_desc->GetName().c_str(), i);
                    return FAILED);
    GELOGD("GenerateGeTensor: raw_data_size %lu", raw_data_size);

    const DataPtrInfo &shape_data_info = data_vec.at(i * kDouble + 1);
    uint64_t shape_data_size = shape_data_info.data_size;
    GELOGD("GenerateGeTensor: shape_data_size %lu", shape_data_size);
    if (shape_data_size == 0) {
      GELOGW("node[%s] outshape is scalar, skip copy shape", node_desc->GetName().c_str());
      output_ptr->MutableTensorDesc().SetShape(GeShape());
      outputs.emplace_back(output_ptr);
      continue;
    }
    uint64_t dim_num = shape_data_size / sizeof(uint64_t);
    std::unique_ptr<int64_t[]> shape_addr(new (std::nothrow) int64_t[dim_num]());
    if (shape_addr == nullptr) {
      REPORT_CALL_ERROR("E19999", "New Buffer failed, size:%lu",
                        dim_num);
      GELOGE(MEMALLOC_FAILED, "[New][Buffer] failed, size:%lu", dim_num);
      return INTERNAL_ERROR;
    }
    GE_CHK_RT_RET(rtMemcpy(shape_addr.get(), shape_data_size,
                           reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(shape_data_info.dst_ptr)),
                           shape_data_size, RT_MEMCPY_DEVICE_TO_HOST));
    std::vector<int64_t> shape_dims;
    for (size_t j = 0; j < dim_num; j++) {
      shape_dims.push_back(shape_addr[j]);
      GELOGD("GenerateGeTensor: dim %ld", shape_addr[j]);
    }
    output_ptr->MutableTensorDesc().SetShape(GeShape(shape_dims));
    outputs.emplace_back(output_ptr);
  }
  return SUCCESS;
}

void AicpuConstantFoldingPass::ReleaseMemory(const vector<AddrAndType> &input_addrs,
                                             const vector<uint64_t> &output_addrs,
                                             const vector<DataPtrInfo> &data_vec) {
  for (const auto &item : input_addrs) {
    GE_CHK_RT(rtFree(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(item.input_addr))));
  }
  for (auto item : output_addrs) {
    GE_CHK_RT(rtFree(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(item))));
  }
  for (const auto &item : data_vec) {
    auto dst_ptr = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(item.dst_ptr));
    if (dst_ptr != nullptr) {
      GE_CHK_RT(rtFree(dst_ptr));
    }
  }
}

bool AicpuConstantFoldingPass::IsSkipFold(const ge::NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  string type = node->GetType();
  if (type == ge::FRAMEWORKOP) {
    if (!ge::AttrUtils::GetStr(node->GetOpDesc(), ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, type)) {
      GELOGW("Skip aicpu constant folding on frameworkop node [%s]", node->GetName().c_str());
      return true;
    }
  }
  auto instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    REPORT_INNER_ERROR("E19999", "GeLib is not init before, check invalid");
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Check][Param] GE is not initialized");
    return true;
  }
  OpsKernelInfoStorePtr kernel_info = instance_ptr->OpsKernelManagerObj().GetOpsKernelInfoStore(kKernelLibName);
  if (kernel_info == nullptr) {
    REPORT_INNER_ERROR("E19999", "Find ops kernel by name:%s failed",
                       kKernelLibName);
    GELOGE(FAILED, "[Get][OpsKernelInfoStore] by name:%s failed", kKernelLibName);
    return true;
  }
  std::string check_result;
  kernel_info->opsFlagCheck(*node, check_result);
  if (check_result.empty()) {
    REPORT_CALL_ERROR("E19999", "Call opsFlagCheck failed, ops kernel name:%s, op:%s(%s)",
                      kKernelLibName, node->GetName().c_str(), node->GetType().c_str());
    GELOGE(FAILED, "[Call][OpsFlagCheck] failed, ops kernel name:%s, op:%s(%s)",
           kKernelLibName, node->GetName().c_str(), node->GetType().c_str());
    return true;
  }
  return check_result.substr(0, kOpsFlag) == kNotSupported;
}
}  // namespace ge
