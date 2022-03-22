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
#include "hybrid/node_executor/controlop/control_op_executor.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/executor/subgraph_executor.h"

namespace ge {
namespace hybrid {
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::CONTROL_OP, ControlOpNodeExecutor);

Status ControlOpNodeTask::ExecuteSubgraph(const GraphItem *subgraph,
                                          TaskContext &task_context,
                                          const std::function<void()> &done_callback) {
  GELOGD("[%s] Start to execute subgraph.", subgraph->GetName().c_str());
  auto execution_context = const_cast<GraphExecutionContext *>(task_context.GetExecutionContext());
  auto executor = MakeShared<SubgraphExecutor>(subgraph, execution_context);
  GE_CHECK_NOTNULL(executor);
  GE_CHK_STATUS_RET(executor->ExecuteAsync(task_context),
                    "[Invoke][ExecuteAsync][%s] Failed to execute partitioned call.", subgraph->GetName().c_str());

  auto callback = [executor, done_callback]() mutable {
    if (done_callback != nullptr) {
      done_callback();
    }
    // executor must outlive task context
    executor.reset();
  };

  GE_CHK_STATUS_RET_NOLOG(task_context.RegisterCallback(callback));
  GELOGD("[%s] Done executing subgraph successfully.", subgraph->GetName().c_str());
  return SUCCESS;
}

Status ControlOpNodeTask::ToBool(const TensorValue &tensor, DataType data_type, bool &value) {
  switch (data_type) {
#define CASE(DT, T)                                       \
  case (DT): {                                            \
    T val{};                                              \
    GE_CHK_STATUS_RET(tensor.CopyScalarValueToHost(val)); \
    value = val != 0;                                     \
    break;                                                \
  }
    // DT_STRING was handled in CondPass
    CASE(DT_FLOAT, float)
    CASE(DT_DOUBLE, double)
    CASE(DT_INT32, int32_t)
    CASE(DT_UINT8, uint8_t)
    CASE(DT_INT16, int16_t)
    CASE(DT_INT8, int8_t)
    CASE(DT_INT64, int64_t)
#undef CASE
    case DT_BOOL:
      GE_CHK_STATUS_RET(tensor.CopyScalarValueToHost(value));
      break;
    default:
      GELOGE(UNSUPPORTED, "Data type %s is not support by cond.", TypeUtils::DataTypeToSerialString(data_type).c_str());
      return UNSUPPORTED;
  }

  return SUCCESS;
}

Status ControlOpNodeTask::UpdateArgs(TaskContext &context) {
  // do nothing
  return SUCCESS;
}

Status ControlOpNodeTask::ExecuteAsync(TaskContext &task_context, std::function<void()> done_callback) {
  auto ret = DoExecuteAsync(task_context, done_callback);
  task_context.SetStatus(ret);
  return ret;
}

Status IfOpNodeTask::Init(const NodePtr &node, const HybridModel &model) {
  GELOGD("[%s] Start to init IfOpNodeTask.", node->GetName().c_str());
  auto then_subgraph = NodeUtils::GetSubgraph(*node, kThenBranchIndex);
  GE_CHECK_NOTNULL(then_subgraph);
  GELOGD("[%s] Adding subgraph [%s] to then-subgraph.", node->GetName().c_str(), then_subgraph->GetName().c_str());
  then_ = model.GetSubgraphItem(then_subgraph);
  GE_CHECK_NOTNULL(then_);

  auto else_subgraph = NodeUtils::GetSubgraph(*node, kElseBranchIndex);
  GE_CHECK_NOTNULL(else_subgraph);
  GELOGD("[%s] Adding subgraph [%s] to else-subgraph.", node->GetName().c_str(), else_subgraph->GetName().c_str());
  else_ = model.GetSubgraphItem(else_subgraph);
  GE_CHECK_NOTNULL(else_);

  GELOGD("[%s] Done initialization successfully.", node->GetName().c_str());
  return SUCCESS;
}

Status IfOpNodeTask::DoExecuteAsync(TaskContext &task_context, const std::function<void()> &done_callback) const {
  auto cond_tensor_desc = task_context.MutableInputDesc(kIfCondIndex);
  GE_CHECK_NOTNULL(cond_tensor_desc);
  auto data_type = cond_tensor_desc->GetDataType();
  const auto &shape = cond_tensor_desc->MutableShape();
  bool cond_val = false;
  if (shape.IsScalar()) {
    auto cond_tensor = task_context.GetInput(kIfCondIndex);
    GE_CHECK_NOTNULL(cond_tensor);
    GE_CHK_STATUS_RET(ToBool(*cond_tensor, data_type, cond_val),
                      "[Invoke][ToBool][%s] Failed to get cond value.",
                      task_context.GetNodeName());
  } else {
    // true if num elements is non-zero
    cond_val = shape.GetShapeSize() != 0;
    GELOGD("[%s] Cond tensor shape = [%s], cond value = %d",
           task_context.GetNodeName(),
           shape.ToString().c_str(),
           cond_val);
  }

  auto subgraph = cond_val ? then_ : else_;
  GELOGD("[%s] Taking subgraph [%s] by cond = [%d]", task_context.GetNodeName(), subgraph->GetName().c_str(), cond_val);
  GE_CHK_STATUS_RET(ExecuteSubgraph(subgraph, task_context, done_callback),
                    "[Execute][Subgraph] failed for [%s]. cond = %d", task_context.GetNodeName(), cond_val);

  GELOGD("[%s] Done executing with cond = %d successfully.", task_context.GetNodeName(), cond_val);
  return SUCCESS;
}

Status CaseOpNodeTask::Init(const NodePtr &node, const HybridModel &model) {
  size_t num_subgraphs = node->GetOpDesc()->GetSubgraphInstanceNames().size();
  GE_CHECK_LE(num_subgraphs, kMaxBranchNum);
  GE_CHECK_GE(num_subgraphs, kMinBranchNum);
  auto num_branches = static_cast<uint32_t>(num_subgraphs);
  GELOGD("[%s] Start to init CaseOpNodeTask with %u branches.", node->GetName().c_str(), num_branches);

  for (uint32_t i = 0; i < num_branches; ++i) {
    auto sub_graph = NodeUtils::GetSubgraph(*node, i);
    GE_CHECK_NOTNULL(sub_graph);
    auto graph_item = model.GetSubgraphItem(sub_graph);
    GE_CHECK_NOTNULL(graph_item);
    GELOGD("[%s] Adding subgraph [%s] to branch %u.", node->GetName().c_str(), sub_graph->GetName().c_str(), i);
    subgraphs_.emplace_back(graph_item);
  }

  GELOGD("[%s] Done initialization successfully.", node->GetName().c_str());
  return SUCCESS;
}

const GraphItem *CaseOpNodeTask::SelectBranch(int32_t branch_index) const {
  // subgraphs_ is non-empty. checked int Init
  if (branch_index < 0 || static_cast<size_t>(branch_index) >= subgraphs_.size()) {
    GELOGI("Branch index out of range. index = %d, num_subgraphs = %zu, will taking last branch.",
           branch_index,
           subgraphs_.size());
    branch_index = subgraphs_.size() - 1;
  }

  return subgraphs_[branch_index];
}

Status CaseOpNodeTask::DoExecuteAsync(TaskContext &task_context, const std::function<void()> &done_callback) const {
  auto branch_tensor = task_context.GetInput(kCaseBranchIndex);
  GE_CHECK_NOTNULL(branch_tensor);
  int32_t branch_index = 0;
  GE_CHK_STATUS_RET(branch_tensor->CopyScalarValueToHost(branch_index));
  const GraphItem *subgraph = SelectBranch(branch_index);
  GELOGI("[%s] Taking subgraph [%s] by branch = [%d]",
         task_context.GetNodeName(),
         subgraph->GetName().c_str(),
         branch_index);

  std::vector<TensorValue> inputs;
  std::vector<TensorValue> outputs;
  for (int i = 0; i < task_context.NumInputs(); ++i) {
    auto input_tensor = task_context.GetInput(i);
    GE_CHECK_NOTNULL(input_tensor);
    inputs.emplace_back(*input_tensor);
  }

  GE_CHK_STATUS_RET(ExecuteSubgraph(subgraph, task_context, done_callback),
                    "[Execute][Subgraph] failed for [%s].", task_context.GetNodeName());

  GELOGD("[%s] Done executing subgraph[%d] successfully.", task_context.GetNodeName(), branch_index);
  return SUCCESS;
}

Status WhileOpNodeTask::Init(const NodePtr &node, const HybridModel &model) {
  GELOGD("[%s] Start to init WhileOpNodeTask.", node->GetName().c_str());
  auto cond_subgraph = NodeUtils::GetSubgraph(*node, kCondBranchIndex);
  GE_CHECK_NOTNULL(cond_subgraph);
  GELOGD("[%s] Adding subgraph [%s] to cond-subgraph.", node->GetName().c_str(), cond_subgraph->GetName().c_str());
  cond_ = model.GetSubgraphItem(cond_subgraph);
  GE_CHECK_NOTNULL(cond_);

  auto body_subgraph = NodeUtils::GetSubgraph(*node, kBodyBranchIndex);
  GE_CHECK_NOTNULL(body_subgraph);
  GELOGD("[%s] Adding subgraph [%s] to body-subgraph.", node->GetName().c_str(), body_subgraph->GetName().c_str());
  body_ = model.GetSubgraphItem(body_subgraph);
  GE_CHECK_NOTNULL(body_);

  GELOGD("[%s] Done initialization successfully.", node->GetName().c_str());
  return SUCCESS;
}

Status WhileOpNodeTask::DoExecuteAsync(TaskContext &task_context, const std::function<void()> &done_callback) const {
  if (task_context.NumInputs() != task_context.NumOutputs()) {
    REPORT_INNER_ERROR("E19999",
                       "[%s] Invalid while args. num_inputs = %d not equal num_outputs = %d",
                       task_context.GetNodeName(), task_context.NumInputs(), task_context.NumOutputs());
    GELOGE(INTERNAL_ERROR,
           "[Check][Param:task_context][%s] Invalid while args. num_inputs = %d, num_outputs = %d",
           task_context.GetNodeName(), task_context.NumInputs(), task_context.NumOutputs());
    return INTERNAL_ERROR;
  }

  bool is_continue = false;
  GE_CHK_STATUS_RET(ExecuteCond(task_context, is_continue),
                    "[Execute][Cond] failed for [%s]", task_context.GetNodeName());
  if (!is_continue) {
    for (int i = 0; i < task_context.NumInputs(); ++i) {
      auto input_tensor = task_context.GetInput(i);
      auto input_tensor_desc = task_context.GetInputDesc(i);
      auto output_tensor_desc = task_context.MutableOutputDesc(i);
      GE_CHECK_NOTNULL(input_tensor);
      GE_CHECK_NOTNULL(input_tensor_desc);
      GE_CHECK_NOTNULL(output_tensor_desc);
      GE_CHK_STATUS_RET_NOLOG(task_context.SetOutput(i, *input_tensor));
      *output_tensor_desc = *input_tensor_desc;
    }

    if (done_callback) {
      done_callback();
    }
    return SUCCESS;
  }

  // backup original input tensor desc
  std::vector<GeTensorDesc> ori_input_desc(task_context.NumInputs());
  for (int i = 0; i < task_context.NumInputs(); ++i) {
    GE_CHK_STATUS_RET_NOLOG(task_context.GetInputDesc(i, ori_input_desc[i]));
  }

  int iteration = 0;
  while (is_continue) {
    ++iteration;
    GELOGD("[%s] Start to execute, iteration = %d", task_context.GetNodeName(), iteration);
    GE_CHK_STATUS_RET(ExecuteOneLoop(task_context, is_continue),
                      "[Invoke][ExecuteOneLoop][%s] Failed to execute iteration %d.",
                      task_context.GetNodeName(), iteration);
  }
  GELOGD("[%s] Quit from loop. current iteration = %d", task_context.GetNodeName(), iteration);
  if (done_callback) {
    done_callback();
  }

  for (int i = 0; i < task_context.NumInputs(); ++i) {
    GE_CHK_STATUS_RET_NOLOG(task_context.UpdateInputDesc(i, ori_input_desc[i]));
  }
  return SUCCESS;
}

Status WhileOpNodeTask::ExecuteCond(TaskContext &task_context, bool &is_continue) const {
  std::vector<TensorValue> inputs;
  std::vector<ConstGeTensorDescPtr> input_desc;
  std::vector<ConstGeTensorDescPtr> output_desc;
  for (int i = 0; i < task_context.NumInputs(); ++i) {
    auto input_tensor = task_context.GetInput(i);
    GE_CHECK_NOTNULL(input_tensor);
    inputs.emplace_back(*input_tensor);
    input_desc.emplace_back(task_context.GetInputDesc(i));
  }

  auto execution_context = const_cast<GraphExecutionContext *>(task_context.GetExecutionContext());
  auto executor = MakeShared<SubgraphExecutor>(cond_, execution_context, task_context.IsForceInferShape());
  GE_CHECK_NOTNULL(executor);
  GELOGD("[%s] Start to execute cond-subgraph.", task_context.GetNodeName());
  GE_CHK_STATUS_RET(executor->ExecuteAsync(inputs, input_desc),
                    "[Invoke][ExecuteAsync] %s Failed to execute partitioned call.", task_context.GetNodeName());
  GELOGD("[%s] Done executing cond-subgraph successfully.", cond_->GetName().c_str());
  GE_CHK_STATUS_RET_NOLOG(task_context.RegisterCallback([executor]() mutable {
    executor.reset();
  }));

  // get cond output
  GE_CHK_STATUS_RET(executor->Synchronize(),
                    "[Invoke][Synchronize][%s] Failed to sync cond-subgraph result.", cond_->GetName().c_str());
  std::vector<TensorValue> cond_outputs;
  std::vector<ConstGeTensorDescPtr> cond_output_desc_list;
  GE_CHK_STATUS_RET(executor->GetOutputs(cond_outputs, cond_output_desc_list),
                    "[Invoke][GetOutputs][%s] Failed to get cond-output.", cond_->GetName().c_str());
  if (cond_outputs.size() != kCondOutputSize || cond_output_desc_list.size() != kCondOutputSize) {
    REPORT_INNER_ERROR("E19999", "[%s] Number of cond outputs(%zu) or size of cond output desc(%zu)"
                       "not equal %zu, check invalid", task_context.GetNodeName(), cond_outputs.size(),
                       cond_output_desc_list.size(), kCondOutputSize);
    GELOGE(INTERNAL_ERROR,
           "[Check][Size][%s] Number of cond outputs(%zu) or Number of cond output desc(%zu) not equal %zu",
           task_context.GetNodeName(), cond_outputs.size(), cond_output_desc_list.size(), kCondOutputSize);
    return INTERNAL_ERROR;
  }

  auto &cond_tensor_desc = cond_output_desc_list[0];
  const auto &shape = cond_tensor_desc->GetShape();
  if (shape.IsScalar()) {
    auto data_type = cond_tensor_desc->GetDataType();
    GE_CHK_STATUS_RET(ToBool(cond_outputs[0], data_type, is_continue),
                      "[Invoke][ToBool][%s] Failed to get cond value.", task_context.GetNodeName());
  } else {
    // true if num elements is non-zero
    is_continue = shape.GetShapeSize() > 0;
    GELOGD("[%s] Cond tensor shape = [%s], is_continue = %d",
           task_context.GetNodeName(),
           shape.ToString().c_str(),
           is_continue);
  }

  return SUCCESS;
}

Status WhileOpNodeTask::MoveOutputs2Inputs(TaskContext &task_context) {
  // set outputs to inputs for next iteration
  for (int i = 0; i < task_context.NumInputs(); ++i) {
    auto input_tensor = task_context.MutableInput(i);
    auto output_tensor = task_context.MutableOutput(i);
    GE_CHECK_NOTNULL(input_tensor);
    GE_CHECK_NOTNULL(output_tensor);
    *input_tensor = *output_tensor;
    output_tensor->Destroy();

    auto input_tensor_desc = task_context.MutableInputDesc(i);
    GE_CHECK_NOTNULL(input_tensor_desc);
    auto output_tensor_desc = task_context.MutableOutputDesc(i);
    GE_CHECK_NOTNULL(output_tensor_desc);
    GELOGD("[%s] To update input shape[%d] by output shape. from [%s] to [%s]",
           task_context.GetNodeName(),
           i,
           input_tensor_desc->GetShape().ToString().c_str(),
           output_tensor_desc->GetShape().ToString().c_str());
    *input_tensor_desc = *output_tensor_desc;
  }

  return SUCCESS;
}

Status WhileOpNodeTask::ExecuteOneLoop(TaskContext &task_context, bool &is_continue) const {
  GELOGD("[%s] Start to execute body-subgraph.", task_context.GetNodeName());
  GE_CHK_STATUS_RET(ExecuteSubgraph(body_, task_context, nullptr),
                    "[Execute][Subgraph] failed for [%s]", task_context.GetNodeName());
  GELOGD("[%s] Done executing body-subgraph successfully.", task_context.GetNodeName());

  // set outputs to inputs for next iteration
  GE_CHK_STATUS_RET(MoveOutputs2Inputs(task_context),
                    "[Move][Outputs2Inputs] failed for [%s]", task_context.GetNodeName());

  GE_CHK_STATUS_RET(ExecuteCond(task_context, is_continue),
                    "[Invoke][ExecuteCond][%s] Failed to execute cond-subgraph", task_context.GetNodeName());

  if (!is_continue) {
    for (int i = 0; i < task_context.NumInputs(); ++i) {
      auto input_desc = task_context.GetInput(i);
      GE_CHECK_NOTNULL(input_desc);
      GE_CHK_STATUS_RET_NOLOG(task_context.SetOutput(i, *input_desc));
    }
  }
  return SUCCESS;
}

Status ControlOpNodeExecutor::LoadTask(const HybridModel &model,
                                       const NodePtr &node,
                                       shared_ptr<NodeTask> &task) const {
  auto node_item = model.GetNodeItem(node);
  GE_CHECK_NOTNULL(node_item);

  std::unique_ptr<ControlOpNodeTask> node_task;
  auto node_type = node->GetType();
  if (node_type == IF || node_type == STATELESSIF) {
    node_task.reset(new(std::nothrow) IfOpNodeTask());
  } else if (node_type == CASE) {
    node_task.reset(new(std::nothrow) CaseOpNodeTask());
  } else if (node_type == WHILE || node_type == STATELESSWHILE) {
    node_task.reset(new(std::nothrow) WhileOpNodeTask());
  } else {
    REPORT_INNER_ERROR("E19999", "[%s] Unsupported type: %s", node->GetName().c_str(), node_type.c_str());
    GELOGE(PARAM_INVALID, "[Check][NodeType][%s] Unsupported type: %s", node->GetName().c_str(), node_type.c_str());
    return PARAM_INVALID;
  }

  GE_CHECK_NOTNULL(node_task);
  GE_CHK_STATUS_RET(node_task->Init(node, model),
                    "[Invoke][Init][%s] Failed to init ControlOpNodeTask.", node->GetName().c_str());

  task = std::move(node_task);
  return SUCCESS;
}

Status ControlOpNodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge