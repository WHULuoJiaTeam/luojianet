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

#include "hybrid/node_executor/host_cpu/host_cpu_node_executor.h"
#include "graph/passes/folding_pass.h"
#include "hybrid/model/hybrid_model.h"
#include "graph/manager/graph_mem_manager.h"
#include "ge_local_engine/engine/host_cpu_engine.h"
#include "aicpu/common/aicpu_task_struct.h"

namespace ge {
namespace hybrid {
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::HOST_CPU, HostCpuNodeExecutor);

Status HostAicpuNodeTask::UpdateArgs(TaskContext &context) {
  if (context.NumInputs() == 0 && context.NumOutputs() == 0) {
    GELOGD("Node[%s] has no input and output, no need to update args.", node_name_.c_str());
    return SUCCESS;
  }

  vector<uint64_t> io_addrs;
  io_addrs.reserve(context.NumInputs() + context.NumOutputs());
  for (int32_t i = 0; i < context.NumInputs(); ++i) {
    auto tensor = context.GetInput(i);
    GE_CHECK_NOTNULL(tensor);
    auto item = MemManager::Instance().HostMemInstance(RT_MEMORY_HBM).GetAlignedPtr(tensor->GetData());
    GE_CHECK_NOTNULL(item.second);
    io_addrs.emplace_back(reinterpret_cast<uintptr_t>(item.second->MutableGet()));
  }

  for (int32_t i = 0; i < context.NumOutputs(); ++i) {
    const auto &output_desc = context.GetOutputDesc(i);
    GE_CHECK_NOTNULL(output_desc);
    AllocationAttr attr;
    attr.SetMemType(HOST_DDR);
    if (context.AllocateOutput(i, *output_desc, nullptr, &attr) != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "node:%s Failed to allocate output %d", context.GetNodeName(), i);
      GELOGE(FAILED, "[Invoke][AllocateOutput]node:%s Failed to allocate output %d", context.GetNodeName(), i);
      return FAILED;
    }
    auto tensor = context.GetOutput(i);
    GE_CHECK_NOTNULL(tensor);
    auto item = MemManager::Instance().HostMemInstance(RT_MEMORY_HBM).GetAlignedPtr(tensor->GetData());
    GE_CHECK_NOTNULL(item.second);
    io_addrs.emplace_back(reinterpret_cast<uintptr_t>(item.second->MutableGet()));
  }
  auto io_addr = args_.get() + sizeof(aicpu::AicpuParamHead);

  // if has input and output, need copy to ioaddr
  int cpy_ret = memcpy_s(io_addr, args_size_ - sizeof(aicpu::AicpuParamHead),
                         &io_addrs[0], sizeof(uint64_t) * io_addrs.size());
  if (cpy_ret != EOK) {
    REPORT_INNER_ERROR("E19999", "Node[%s] memcpy io addr to AicpuParamHead failed,"
                       "ret=%d, args_size=%u, io nums=%zu.",
                       node_name_.c_str(), cpy_ret, args_size_, io_addrs.size());
    GELOGE(INTERNAL_ERROR, "[Update][io_addr]Node[%s] memcpy io addr to AicpuParamHead failed,"
           "ret=%d, args_size=%u, io nums=%zu.",
           node_name_.c_str(), cpy_ret, args_size_, io_addrs.size());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status HostAicpuNodeTask::ExecuteAsync(TaskContext &context, std::function<void()> done_callback) {
  GELOGD("[%s] Start execute.", context.GetNodeName());
  GE_CHK_STATUS_RET(Execute(context), "[Invoke][Execute] failed for node:%s.", node_name_.c_str());
  if (done_callback) {
    GELOGD("[%s] Start invoke callback.", context.GetNodeName());
    done_callback();
  }
  GELOGD("[%s] Done execute successfully.", context.GetNodeName());
  return SUCCESS;
}

Status HostAicpuNodeTask::Execute(TaskContext &context) {
  GELOGD("Node[%s] launch task start.", node_name_.c_str());
  if (run_cpu_kernel_) {
    GE_CHK_STATUS_RET(run_cpu_kernel_(args_.get()), "[Run][CpuKernel] failed for node:%s.", node_name_.c_str());
  } else {
    REPORT_CALL_ERROR("E19999", "Run cpu kernel failed node:%s, cpu kernel is not initialized.", node_name_.c_str());
    GELOGE(INTERNAL_ERROR,
           "[Run][Kernel]Run cpu kernel failed node:%s, cpu kernel is not initialized.", node_name_.c_str());
    return INTERNAL_ERROR;
  }

  GELOGD("Node[%s] launch task successfully.", node_name_.c_str());
  return SUCCESS;
}

Status HostAicpuNodeTask::SetHostExtInfo() {
  if (aicpu_ext_handle_.GetExtInfoLen() == 0) {
    GELOGD("Node[%s] don't have ext info, no need update.", node_name_.c_str());
    return SUCCESS;
  }

  auto aicpu_param_head = reinterpret_cast<aicpu::AicpuParamHead *>(args_.get());
  GE_CHECK_NOTNULL(aicpu_param_head);
  aicpu_param_head->extInfoLength = aicpu_ext_handle_.GetExtInfoLen();
  aicpu_param_head->extInfoAddr = reinterpret_cast<uintptr_t>(aicpu_ext_handle_.GetExtInfo());
  return SUCCESS;
}

Status HostCpuNodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  return task.UpdateArgs(context);
}

Status HostCpuNodeExecutor::ValidateTaskDef(const domi::TaskDef &task_def) {
  auto task_type = static_cast<rtModelTaskType_t>(task_def.type());
  if (task_type != RT_MODEL_TASK_KERNEL) {
    REPORT_CALL_ERROR("E19999", "[Check][TaskType]Invalid task type (%d) in host cpu excutor.",
                      static_cast<int>(task_type));
    GELOGE(INTERNAL_ERROR,
           "[Check][TaskType]Invalid task type (%d) in host cpu excutor.", static_cast<int>(task_type));
    return INTERNAL_ERROR;
  }
  auto kernel_type = static_cast<ccKernelType>(task_def.kernel().context().kernel_type());
  if (kernel_type != ccKernelType::HOST_CPU) {
    REPORT_INNER_ERROR("E19999", "Invalid kernel type(%d) in host cpu excutor.",
                       static_cast<int>(kernel_type));
    GELOGE(INTERNAL_ERROR,
           "[Check][TaskType]Invalid kernel type(%d) in host cpu excutor.", static_cast<int>(kernel_type));
    return INTERNAL_ERROR;
  }

  return SUCCESS;
}

Status HostCpuNodeExecutor::LoadTask(const HybridModel &model, const NodePtr &node,
                                     std::shared_ptr<NodeTask> &task) const {
  GE_CHECK_NOTNULL(node);
  auto node_item = model.GetNodeItem(node);
  GE_CHECK_NOTNULL(node_item);
  auto task_defs = model.GetTaskDefs(node);
  GE_CHECK_NOTNULL(task_defs);

  if ((*task_defs).size() != 1) {
    REPORT_CALL_ERROR("E19999", "[Check][Size]Node[%s] task_def num[%zu] != 1",
                      node->GetName().c_str(), (*task_defs).size());
    GELOGE(PARAM_INVALID, "[Check][Size]Node[%s] task_def num[%zu] != 1",
           node->GetName().c_str(), (*task_defs).size());
    return PARAM_INVALID;
  }
  const auto &task_def = (*task_defs)[0];
  GE_CHK_STATUS_RET(ValidateTaskDef(task_def),
                    "[Validate][TaskDef] failed for Node[%s].", node->GetName().c_str());
  auto host_aicpu_task = MakeShared<HostAicpuNodeTask>(node_item, task_def);
  GE_CHK_BOOL_RET_STATUS(host_aicpu_task != nullptr, MEMALLOC_FAILED,
                         "[Check][State]Load task for node %s failed.", node->GetName().c_str());
  GE_CHK_STATUS_RET(host_aicpu_task->Init(model),
                    "[Init][AicpuNodeTaskBase] failed for Node[%s].", node->GetName().c_str());
  GE_CHK_STATUS_RET(host_aicpu_task->SetHostExtInfo(),
                    "[Set][HostExtInfo] failed for Node[%s].", node->GetName().c_str());

  auto handle = HostCpuEngine::GetInstance().GetConstantFoldingHandle();
  if (handle == nullptr) {
    REPORT_CALL_ERROR("E19999", "Get constant folding handle failed.");
    GELOGE(INTERNAL_ERROR, "[Get][Handle]Get constant folding handle failed.");
    return INTERNAL_ERROR;
  }
  auto run_cpu_kernel = (uint32_t (*)(void *))mmDlsym(handle, "RunHostCpuKernel");
  if (run_cpu_kernel != nullptr) {
    host_aicpu_task->SetRunKernel(run_cpu_kernel);
  } else {
    REPORT_CALL_ERROR("E19999", "Get run cpu kernel failed.");
    GELOGE(INTERNAL_ERROR, "[Get][Kernel]Get run cpu kernel failed.");
    return INTERNAL_ERROR;
  }

  task = std::move(host_aicpu_task);
  GELOGD("Node[%s] load task end.", node->GetName().c_str());

  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
