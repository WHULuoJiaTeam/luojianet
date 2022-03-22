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

#include "hybrid/node_executor/aicore/aicore_node_executor.h"
#include "framework/common/taskdown_common.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "external/runtime/rt_error_codes.h"
#include "single_op/task/build_task_utils.h"

namespace ge {
namespace hybrid {
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::AICORE, AiCoreNodeExecutor);
namespace {
bool IsNoOp(const NodeItem &node_item) {
  for (int i = 0; i < node_item.num_outputs; ++i) {
    const auto &tensor_desc = node_item.MutableOutputDesc(i);
    GE_CHECK_NOTNULL(tensor_desc);
    const auto &shape = tensor_desc->MutableShape();
    if (shape.IsScalar() || shape.GetShapeSize() > 0) {
      return false;
    }
  }

  return true;
}
}  // namespace
AiCoreNodeTask::AiCoreNodeTask(std::vector<std::unique_ptr<AiCoreOpTask>> &&tasks) : tasks_(std::move(tasks)) {
}

Status AiCoreNodeExecutor::Initialize() {
  compiler_ = TaskCompilerFactory::GetInstance().GetTaskCompiler();
  return SUCCESS;
}

Status AiCoreNodeExecutor::LoadTask(const HybridModel &model, const NodePtr &node, shared_ptr<NodeTask> &task) const {
  GE_CHECK_NOTNULL(node);
  GELOGI("AiCoreNodeExecutor(%s) LoadTask Start.", node->GetName().c_str());
  bool is_single_op = model.IsSingleOp();

  auto *task_defs = model.GetTaskDefs(node);
  if (task_defs == nullptr || task_defs->empty()) {
    bool dynamic_flag = false;
    if (!AttrUtils::GetBool(node->GetOpDesc(), "support_dynamicshape", dynamic_flag) || !dynamic_flag) {
      GELOGD("Skip create task of node (%s) as 'support_dynamicshape' is false and cann't get task_defs.",
             node->GetName().c_str());
      return SUCCESS;
    } else {
      GELOGE(FAILED, "[Invoke][GetBool]Task_defs is empty for node (%s)"
             "which 'support_dynamicshape' is true, check invalid",
             node->GetName().c_str());
      REPORT_CALL_ERROR("E19999", "Task_defs is empty for node (%s)"
                        "which 'support_dynamicshape' is true, check invalid",
                        node->GetName().c_str());
      return FAILED;
    }
  }

  AiCoreTaskBuilder builder(node->GetOpDesc(), *task_defs);
  std::unique_ptr<AiCoreNodeTask> node_task;
  GE_CHK_STATUS_RET(builder.BuildTask(node_task, true, is_single_op),
                    "[Invoke][BuildTask][%s] Failed to build op tasks.", node->GetName().c_str());
  task = std::move(node_task);
  GELOGI("AiCoreNodeExecutor(%s) LoadTask End.", node->GetName().c_str());
  return SUCCESS;
}

Status AiCoreNodeExecutor::GenNodeKey(const NodePtr &node, std::string &node_key) {
  GE_CHECK_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  // make sure unique, (op_id + input_shape) is unique
  node_key = std::to_string(op_desc->GetId()) + "-";
  node_key.append(std::to_string(op_desc->GetInputsSize()));
  auto input_descs = op_desc->GetAllInputsDescPtr();
  for (auto &input_desc : input_descs) {
    node_key.push_back('-');
    auto &shape = input_desc->MutableShape();
    auto num_dims = shape.GetDimNum();
    if (num_dims == 0) {
      continue;
    } // scalar
    for (std::size_t i = 0; i < num_dims - 1; i++) {
      node_key.append(std::to_string(shape.GetDim(i)));
      node_key.push_back('_');
    }
    node_key.append(std::to_string(shape.GetDim(num_dims - 1)));
  }
  return SUCCESS;
}

bool AiCoreNodeTaskRegistry::AddTask(const std::string &node_key, const std::shared_ptr<AiCoreNodeTask> &task) {
  GE_CHECK_NOTNULL(task);
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = reg_node_tasks_.find(node_key);
  if (iter != reg_node_tasks_.end()) {
    GELOGE(FAILED, "[Add][Task] failed, key:%s already exist.", node_key.c_str());
    REPORT_INNER_ERROR("E19999", "AddTask failed, key:%s already exist.", node_key.c_str());
    return false;
  }
  auto ret = reg_node_tasks_.emplace(node_key, task);
  return ret.second;
}

std::shared_ptr<AiCoreNodeTask> AiCoreNodeTaskRegistry::GetTask(const std::string &node_key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = reg_node_tasks_.find(node_key);
  return (iter != reg_node_tasks_.end()) ? iter->second : nullptr;
}

Status AiCoreNodeExecutor::CompileTask(const HybridModel &model,
                                       const NodePtr &node, shared_ptr<NodeTask> &task) const {
  auto node_item = model.GetNodeItem(node);
  GE_CHECK_NOTNULL(node_item);
  if (IsNoOp(*node_item)) {
    task = MakeShared<NoOpTask>();
    return SUCCESS;
  }
  auto op_desc = node_item->op_desc;
  GELOGI("AiCoreNodeExecutor(%s) CompileTask Start.", node->GetName().c_str());

  auto ori_node_name = node->GetName();
  if (compiler_ == nullptr) {
    GELOGE(FAILED, "[Find][Compiler][%s] Can not find any valid aicore task compiler.", ori_node_name.c_str());
    REPORT_INNER_ERROR("E19999", "[%s] Can not find any valid aicore task compiler.", ori_node_name.c_str());
    return FAILED;
  }

  AiCoreNodeTaskRegistry &registry = AiCoreNodeTaskRegistry::GetInstance();
  std::string shape_key;
  GE_CHK_STATUS_RET(GenNodeKey(node, shape_key), "[Generate][NodeKey] failed, op name = %s.", node->GetName().c_str());

  auto node_key = std::to_string(model.GetModelId()) + "/" + shape_key;
  GELOGD("NodeKey for %s = %s", node->GetName().c_str(), node_key.c_str());
  auto aicore_task = registry.GetTask(node_key);
  if (aicore_task != nullptr) {
    // The workspaces needed by a operator may differ with different shapes
    op_desc->SetWorkspaceBytes(aicore_task->GetWorkspaceSizes());
    GELOGI("AiCoreNodeExecutor(%s) CompileTask Skip.", node->GetName().c_str());
    task = std::move(aicore_task);
    return SUCCESS;
  }

  std::vector<domi::TaskDef> task_defs;
  op_desc->SetName(ori_node_name + "_" + shape_key);
  GE_CHK_STATUS_RET(compiler_->CompileOp(node, task_defs), "[Compile][Op:%s] failed.", ori_node_name.c_str());
  op_desc->SetName(ori_node_name);
  GELOGD("successfully generated task_defs: %s", node->GetName().c_str());

  AiCoreTaskBuilder builder(node->GetOpDesc(), task_defs);
  std::unique_ptr<AiCoreNodeTask> node_task;
  GE_CHK_STATUS_RET(builder.BuildTask(node_task, false),
                    "[Invoke][BuildTask][%s] Failed to build op tasks.", node->GetName().c_str());
  node_task->SetWorkspaceSizes(op_desc->GetWorkspaceBytes());
  aicore_task = std::move(node_task);
  GELOGD("successfully created node task: %s", node->GetName().c_str());

  if (!registry.AddTask(node_key, aicore_task)) {
    GELOGE(INTERNAL_ERROR, "[Add][NodeTask] failed, op name = %s.", node->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "add task failed, op name = %s.", node->GetName().c_str());
    return INTERNAL_ERROR;
  }

  task = std::move(aicore_task);
  GELOGI("AiCoreNodeExecutor(%s) CompileTask End.", node->GetName().c_str());
  return SUCCESS;
}

Status AiCoreNodeTask::ExecuteAsync(TaskContext &context, std::function<void()> done_callback) {
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeTaskExecuteAsync] Start");
  if (IsNoOp(context.GetNodeItem())) {
    GELOGD("[%s] Skipping execution for op with empty outputs", context.GetNodeName());
    auto ret = context.TryExecuteCallback(done_callback);
    RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeTaskExecuteAsync] End");
    return ret;
  }

  GELOGI("[%s] ExecuteAsync Start.", context.GetNodeName());
  for (auto it = tasks_.begin(); it != tasks_.end(); ++it) {
    // AtomicAddrClean has 2 tasks
    if (tasks_.size() == 2 && it == tasks_.begin() && !(*(tasks_.rbegin()))->GetClearAtomic()) {
      continue;
    }
    RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeLaunchKernel] Start");
    GE_CHK_STATUS_RET_NOLOG((*it)->LaunchKernel(context.GetStream()));
    GE_CHK_STATUS_RET_NOLOG(CheckOverflow(context));
    GE_CHECK_NOTNULL(context.GetExecutionContext()->model);
    GELOGD("[DEBUG_TASK_INFO : Executor Task] %s/%s %s",
           context.GetExecutionContext()->model->GetModelName().c_str(),
           (*it)->GetName().empty() ? (*it)->GetLogName().c_str() : (*it)->GetName().c_str(),
           BuildTaskUtils::GetTaskInfo(context).c_str());
    // save profiling data
    uint32_t task_id = 0;
    uint32_t stream_id = 0;
    rtError_t rt_ret = rtGetTaskIdAndStreamID(&task_id, &stream_id); // must be called after Launch kernel
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "[Invoke][rtGetTaskIdAndStreamID] failed, ret: 0x%X.", rt_ret);
      REPORT_CALL_ERROR("E19999", "rtGetTaskIdAndStreamID failed, ret: 0x%X.", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
    context.SetTaskId(task_id);
    context.SetStreamId(stream_id);
    GELOGD("Aicore node[%s] task_id: %u, stream_id: %u.", context.GetNodeName(), task_id, stream_id);
    (void)context.SaveProfilingTaskDescInfo(task_id, stream_id, kTaskTypeAicore, (*it)->GetBlockDim(), (*it)->GetOpType());
    RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeLaunchKernel] End");
    RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeLaunchKernel] End");
  }

  if (done_callback != nullptr) {
    RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeRegisterCallback] Start");
    GE_CHK_STATUS_RET_NOLOG(context.RegisterCallback(done_callback));
    RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeRegisterCallback] End");
  }

  GELOGD("[%s] ExecuteAsync End.", context.GetNodeName());
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeTaskExecuteAsync] End");
  return SUCCESS;
}

Status AiCoreNodeTask::UpdateArgs(TaskContext &context) {
  GELOGI("[%s] AiCoreNodeTask UpdateArgs Start.", context.GetNodeName());
  for (auto it = tasks_.rbegin(); it != tasks_.rend(); ++it) {
    GE_CHK_STATUS_RET_NOLOG((*it)->UpdateArgs(context));
    // AtomicAddrClean has 2 tasks
    if (tasks_.size() == 2 && it == tasks_.rbegin() && !(*it)->GetClearAtomic()) {
      break;
    }
  }
  GELOGI("[%s] AiCoreNodeTask UpdateArgs End.", context.GetNodeName());
  return SUCCESS;
}

Status AiCoreNodeTask::UpdateTilingData(TaskContext &context) {
  GELOGD("[%s] PrepareWithShape started", context.GetNodeName());
  for (auto it = tasks_.rbegin(); it != tasks_.rend(); ++it) {
    GE_CHK_STATUS_RET_NOLOG((*it)->PrepareWithShape(context));
    // AtomicAddrClean has 2 tasks
    if (tasks_.size() == 2 && it == tasks_.rbegin() && !(*it)->GetClearAtomic()) {
      break;
    }
  }
  GELOGD("[%s] Done PrepareWithShape successfully.", context.GetNodeName());
  return SUCCESS;
}

bool AiCoreNodeTask::IsSupportDynamicShape() {
  for (size_t i = 0; i < tasks_.size(); ++i) {
    if (!tasks_[i]->IsDynamicShapeSupported()) {
      GELOGD("[%s] Task does not support dynamic shape.", tasks_[i]->GetName().c_str());
      return false;
    }
  }

  return true;
}

const vector<int64_t> &AiCoreNodeTask::GetWorkspaceSizes() const {
  return workspace_sizes_;
}

void AiCoreNodeTask::SetWorkspaceSizes(const vector<int64_t> &workspace_sizes) {
  workspace_sizes_ = workspace_sizes;
}

Status AiCoreNodeTask::CheckOverflow(TaskContext &context) {
  const DumpProperties &dump_properties = context.GetDumpProperties();
  if (dump_properties.IsOpDebugOpen()) {
    GELOGD("Op %s is doing overflow check in hybrid engine", context.GetNodeName());
    auto rt_ret = rtStreamSynchronize(context.GetStream());
    if (rt_ret == ACL_ERROR_RT_AICORE_OVER_FLOW) {
      context.SetOverFlow(true);
      GELOGW("Dynamic shape op %s is over flow", context.GetNodeName());
      return SUCCESS;
    } else if (rt_ret != RT_ERROR_NONE) {
      GELOGE(rt_ret, "[Invoke][rtstreamsynchronize] failed, ret:%d.", rt_ret);
      REPORT_CALL_ERROR("E19999", "rtstreamsynchronize failed, ret:%d.", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
    return SUCCESS;
  }
  GELOGD("Opdebug is not open in hybrid engine");
  return SUCCESS;
}

TaskCompilerFactory &TaskCompilerFactory::GetInstance() {
  static TaskCompilerFactory instance;
  return instance;
}

void TaskCompilerFactory::Register(CreateFn fn) {
  compiler_func_ = fn;
}

std::unique_ptr<TaskCompiler> TaskCompilerFactory::GetTaskCompiler() {
  if (compiler_func_ == nullptr) {
    return nullptr;
  }
  auto compiler_instance = std::unique_ptr<TaskCompiler>(compiler_func_());
  return compiler_instance;
}

CompilerFunctionRegistrar::CompilerFunctionRegistrar(CreateFn fn) {
  TaskCompilerFactory::GetInstance().Register(fn);
}
}  // namespace hybrid
}  // namespace ge
