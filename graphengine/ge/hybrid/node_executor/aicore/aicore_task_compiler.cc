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

#include "hybrid/node_executor/aicore/aicore_task_compiler.h"
#include "framework/common/debug/log.h"
#include "graph/debug/ge_attr_define.h"
#include "opskernel_manager/ops_kernel_builder_manager.h"
#include "init/gelib.h"

namespace ge {
namespace hybrid {
namespace {
uintptr_t kWeightBase = 0x10000000;
uintptr_t kMemBase = 0x20000000;
uint64_t kFakeSize = 0x10000000UL;
REGISTER_TASK_COMPILER(AiCoreTaskCompiler);
}
std::mutex AiCoreTaskCompiler::mu_;

Status AiCoreTaskCompiler::Initialize() {
  std::lock_guard<std::mutex> lk(mu_);
  if (is_initialized_) {
    return SUCCESS;
  }

  auto ge_lib = GELib::GetInstance();
  GE_CHECK_NOTNULL(ge_lib);
  if (!ge_lib->InitFlag()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Check][State] failed, because Ge_lib is uninitialized.");
    REPORT_INNER_ERROR("E19999", "Initialize failed, because Ge_lib is uninitialized.");
    return GE_CLI_GE_NOT_INITIALIZED;
  }
  auto &kernel_manager = ge_lib->OpsKernelManagerObj();
  aic_kernel_store_ = kernel_manager.GetOpsKernelInfoStore("AIcoreEngine");
  GE_CHECK_NOTNULL(aic_kernel_store_);
  is_initialized_ = true;
  return SUCCESS;
}

Status AiCoreTaskCompiler::DoCompileOp(const NodePtr &node) const {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(aic_kernel_store_);
  vector<NodePtr> node_vec;
  node_vec.emplace_back(node);
  GE_CHK_STATUS_RET(aic_kernel_store_->CompileOpRun(node_vec),
                    "[Invoke][CompileOpRun] Failed, node = %s", node->GetName().c_str());
  GE_CHK_STATUS_RET(OpsKernelBuilderManager::Instance().CalcOpRunningParam(*node),
                    "[Invoke][CalcOpRunningParam] Failed, node = %s", node->GetName().c_str());
  return SUCCESS;
}

Status AiCoreTaskCompiler::CompileOp(const NodePtr &node, std::vector<domi::TaskDef> &tasks) {
  Status ret = Initialize();
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Check][State][%s] Offline inference not support online compile.", node->GetName().c_str());
    REPORT_INNER_ERROR("E19999", "[%s] Offline inference not support online compile.", node->GetName().c_str());
    return ret;
  }

  GE_CHECK_NOTNULL(node);
  GELOGI("AiCoreTaskCompiler(%s) CompileOp Start.", node->GetName().c_str());

  auto op_desc = node->GetOpDesc();
  op_desc->SetWorkspaceBytes({});
  GE_CHK_STATUS_RET_NOLOG(DoCompileOp(node));
  GELOGD("successfully compiled op: %s", node->GetName().c_str());

  std::vector<int64_t> input_offsets(op_desc->GetInputsSize(), kMemBase);
  std::vector<int64_t> output_offsets(op_desc->GetOutputsSize(), kMemBase);
  op_desc->SetInputOffset(input_offsets);
  op_desc->SetOutputOffset(output_offsets);
  std::vector<int64_t> workspaces(op_desc->GetWorkspaceBytes().size(), kMemBase);
  op_desc->SetWorkspace(std::move(workspaces));
  GE_CHK_STATUS_RET_NOLOG(DoGenerateTask(*node, tasks));
  GELOGD("successfully generated task: %s", node->GetName().c_str());
  GELOGI("AiCoreTaskCompiler(%s) CompileOp End.", node->GetName().c_str());
  return SUCCESS;
}

Status AiCoreTaskCompiler::DoGenerateTask(const Node &node,
                                          std::vector<domi::TaskDef> &tasks) {
  rtModel_t rt_model_ = nullptr;
  GE_CHK_RT_RET(rtModelCreate(&rt_model_, 0));
  rtStream_t stream = nullptr;
  GE_CHK_RT_EXEC(rtStreamCreate(&stream, 0), GE_CHK_RT(rtModelDestroy(rt_model_)); return RT_FAILED);
  GE_MAKE_GUARD_RTSTREAM(stream);
  GE_CHK_RT_EXEC(rtModelBindStream(rt_model_, stream, 0), GE_CHK_RT(rtModelDestroy(rt_model_)); return RT_FAILED);

  RunContext context;
  context.stream = stream;
  context.model = rt_model_;
  context.graphStreamList.emplace_back(stream);
  context.weightMemBase = reinterpret_cast<uint8_t *>(kWeightBase);
  context.dataMemBase = reinterpret_cast<uint8_t *>(kWeightBase);
  context.weightMemSize = kFakeSize;
  context.dataMemSize = kFakeSize;

  Status ret;
  {
    std::lock_guard<std::mutex> lk(mu_);
    ret = OpsKernelBuilderManager::Instance().GenerateTask(node, context, tasks);
  }

  GE_CHK_STATUS(ret, "[Invoke][GenerateTask] Failed, node = %s", node.GetName().c_str());
  GE_CHK_RT(rtModelUnbindStream(rt_model_, stream));
  GE_CHK_RT(rtModelDestroy(rt_model_));
  return ret;
}
}  // namespace hybrid
}  // namespace ge
