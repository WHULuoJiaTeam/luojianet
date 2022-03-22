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

#include "hybrid/node_executor/ge_local/ge_local_node_executor.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/util.h"
#include "hybrid/model/hybrid_model.h"
#include "inc/kernel.h"
#include "inc/kernel_factory.h"
#include "common/ge/ge_util.h"
#include "hybrid/executor/hybrid_execution_context.h"

namespace ge {
namespace hybrid {
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::GE_LOCAL, GeLocalNodeExecutor);

const std::map<std::string, std::vector<uint32_t>>
    RefInputTask::out_ref_input_index_ = {{DATA, {}},
                                          {AIPPDATA, {}},
                                          {RESHAPE, {}},
                                          {EXPANDDIMS, {}},
                                          {SQUEEZE, {}},
                                          {UNSQUEEZE, {}},
                                          {BROADCASTGRADIENTARGS, {}}
                                         };

const std::set<std::string> DependInputShapeTask::depend_input_shape_ops_ = {SHAPE, SHAPEN, RANK, SIZE};

const std::set<std::string> ConstantNodeTask::constant_like_task_ops_ = {CONSTANT, CONSTANTOP, VARIABLE};

const std::set<std::string> NoOpNodeTask::control_only_task_ops_ = {NOOP, CONTROLTRIGGER};

Status RefInputTask::UpdateArgs(TaskContext &) {
  // no need update args
  return SUCCESS;
}

Status RefInputTask::Execute(TaskContext &context) {
  auto iter = out_ref_input_index_.find(node_type_);
  if (iter == out_ref_input_index_.end()) {
    REPORT_INNER_ERROR("E19999", "node %s type %s can not use RefInputTask.",
                       node_name_.c_str(), node_type_.c_str());
    GELOGE(UNSUPPORTED, "[Find][Node]node %s type %s can not use RefInputTask.",
           node_name_.c_str(), node_type_.c_str());
    return UNSUPPORTED;
  }

  auto &ref_index = iter->second;
  if (ref_index.empty()) {
    return RefOneByOne(context);
  } else {
    return RefByOrder(ref_index, context);
  }
}

Status RefInputTask::RefOneByOne(TaskContext &context) {
  GELOGI("node %s type %s ref input one by one begin.", node_name_.c_str(), node_type_.c_str());
  int input_num = context.NumInputs();
  int output_num = context.NumOutputs();
  if (output_num > input_num) {
    REPORT_INNER_ERROR("E19999", "node %s type %s has %d outputs but only %d inputs, can't ref one by one.",
                       node_name_.c_str(), node_type_.c_str(), output_num, input_num);
    GELOGE(INTERNAL_ERROR, "[Check][Size]node %s type %s has %d outputs but only %d inputs, can't ref one by one.",
           node_name_.c_str(), node_type_.c_str(), output_num, input_num);
    return INTERNAL_ERROR;
  }
  for (uint32_t out_index = 0; out_index < static_cast<uint32_t>(output_num); ++out_index) {
    auto input = context.GetInput(out_index);
    GE_CHECK_NOTNULL(input);
    GE_CHK_STATUS_RET(context.SetOutput(out_index, *input));
    GELOGD("node %s type %s output[%d] ref input[%d] addr=%p.",
           node_name_.c_str(), node_type_.c_str(), out_index, out_index, input->GetData());
  }
  GELOGI("node %s type %s ref input one by one end.", node_name_.c_str(), node_type_.c_str());
  return SUCCESS;
}

Status RefInputTask::RefByOrder(const std::vector<uint32_t> &ref_order, TaskContext &context) {
  GELOGI("node %s type %s ref input by order begin.", node_name_.c_str(), node_type_.c_str());
  int32_t output_num = context.NumOutputs();
  if (ref_order.size() != static_cast<size_t>(output_num)) {
    REPORT_INNER_ERROR("E19999", "node %s type %s has %d outputs but only has %zu out ref index.",
                       node_name_.c_str(), node_type_.c_str(), output_num, ref_order.size());
    GELOGE(INTERNAL_ERROR, "[Check][Size]node %s type %s has %d outputs but only has %zu out ref index.",
           node_name_.c_str(), node_type_.c_str(), output_num, ref_order.size());
    return INTERNAL_ERROR;
  }
  for (auto out_index = 0; out_index < output_num; ++out_index) {
    auto ref_input_index = ref_order[out_index];
    auto input = context.GetInput(ref_input_index);
    GE_CHECK_NOTNULL(input);
    GE_CHK_STATUS_RET(context.SetOutput(out_index, *input));
    GELOGD("node %s type %s output[%d] ref input[%u] addr=%p.",
           node_name_.c_str(), node_type_.c_str(), out_index, ref_input_index, input->GetData());
  }
  GELOGI("node %s type %s ref input by order end.", node_name_.c_str(), node_type_.c_str());
  return SUCCESS;
}

Status RefInputTask::ExecuteAsync(TaskContext &context, std::function<void()> done_callback) {
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[RefInputTaskExecuteAsync] Start");
  GE_CHK_STATUS_RET(Execute(context), "[Invoke][Execute]node:%s type:%s ref input task execute failed",
                    node_name_.c_str(), node_type_.c_str());
  if (done_callback != nullptr) {
    // host cpu no need register callback, call it directly.
    GE_CHK_STATUS_RET(context.TryExecuteCallback(done_callback));
  }
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[RefInputTaskExecuteAsync] End");
  return SUCCESS;
}

bool RefInputTask::IsBelong(const std::string &op_type) {
  return out_ref_input_index_.count(op_type) > 0;
}

Status DependInputShapeTask::UpdateArgs(TaskContext &) {
  // no need update args
  return SUCCESS;
}

Status DependInputShapeTask::Execute(TaskContext &context) {
  KernelFactory &factory = KernelFactory::Instance();
  std::string node_type = node_->GetType();
  auto kernel = factory.Create(node_type);
  if (kernel == nullptr) {
    REPORT_CALL_ERROR("E19999", "create failed for node %s type %s is not supported by host kernel.",
                      node_->GetName().c_str(), node_type.c_str());
    GELOGE(UNSUPPORTED, "[Invoke][Create]node %s type %s is not supported by host kernel.",
           node_->GetName().c_str(), node_type.c_str());
    return UNSUPPORTED;
  }
  std::vector<GeTensorPtr> outputs;
  Status compute_ret = kernel->Compute(node_, outputs);
  if (compute_ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "node %s type %s compute failed.", node_->GetName().c_str(), node_type.c_str());
    GELOGE(compute_ret, "[Invoke][Compute]node %s type %s compute failed or not imply.",
           node_->GetName().c_str(), node_type.c_str());
    return compute_ret;
  }
  int32_t output_num = context.NumOutputs();
  if (static_cast<size_t>(output_num) != outputs.size()) {
    REPORT_INNER_ERROR("E19999", "node %s type %s has %d output,"
                       "but kernel compute only has %zu output. check invalid",
                       node_->GetName().c_str(), node_type.c_str(), output_num, outputs.size());
    GELOGE(INTERNAL_ERROR, "[Check][Size]node %s type %s has %d output, but kernel compute only has %zu output.",
           node_->GetName().c_str(), node_type.c_str(), output_num, outputs.size());
    return INTERNAL_ERROR;
  }

  // alloc output
  GE_CHK_STATUS_RET_NOLOG(context.AllocateOutputs(NpuMemoryAllocator::AttrWithDefaultPadding()));

  // copy data to output
  for (auto i = 0; i < output_num; ++i) {
    GeTensorPtr &tensor = outputs[i];
    GE_CHECK_NOTNULL(tensor);
    auto tensor_data = tensor->GetData();
    auto tensor_value = context.MutableOutput(i);
    GE_CHECK_NOTNULL(tensor_value);
    if (tensor_data.GetSize() > tensor_value->GetSize()) {
      REPORT_INNER_ERROR("E19999", "node:%s type:%s [%d]th compute data size=%zu, but context data size=%zu."
                         "check invalid",
                         node_->GetName().c_str(), node_type.c_str(), i,
                         tensor_data.GetSize(), tensor_value->GetSize());
      GELOGE(INTERNAL_ERROR, "[Check][Size]node:%s type:%s [%d]th compute data size=%zu, but context data size=%zu.",
             node_->GetName().c_str(), node_type.c_str(), i, tensor_data.GetSize(), tensor_value->GetSize());
      return INTERNAL_ERROR;
    }

    GELOGI("node:%s type:%s [%d]th output data=%p, out size=%zu, data size=%zu.",
           node_->GetName().c_str(), node_type.c_str(), i,
           tensor_value->GetData(), tensor_value->GetSize(), tensor_data.GetSize());

    if (tensor_data.GetSize() > 0) {
      GE_CHK_RT_RET(rtMemcpy(tensor_value->MutableData(),
                             tensor_value->GetSize(),
                             tensor_data.GetData(),
                             tensor_data.GetSize(),
                             RT_MEMCPY_HOST_TO_DEVICE));
    }
    GELOGI("node:%s type:%s [%d]th set data success, data size=%zu.",
           node_->GetName().c_str(), node_type.c_str(), i, tensor_data.GetSize());
  }
  return SUCCESS;
}

Status DependInputShapeTask::ExecuteAsync(TaskContext &context, std::function<void()> done_callback) {
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(),
                         "[DependInputShapeTaskExecuteAsync] Start");
  GE_CHK_STATUS_RET(Execute(context), "[Invoke][Execute]node:%s type:%s depend input shape task execute failed",
                    node_->GetName().c_str(), node_->GetType().c_str());
  if (done_callback != nullptr) {
    // host cpu no need register callback, call it directly.
    GE_CHK_STATUS_RET(context.TryExecuteCallback(done_callback));
  }
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(),
                         "[DependInputShapeTaskExecuteAsync] End");
  return SUCCESS;
}

bool DependInputShapeTask::IsBelong(const std::string &op_type) {
  return depend_input_shape_ops_.count(op_type) > 0;
}

Status GeLocalNodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(),
                         "[GeLocalNodeExecutorPrepareTask] Start");
  Status status = task.UpdateArgs(context);
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[GeLocalNodeExecutorPrepareTask] End");
  return status;
}

Status GeLocalNodeExecutor::LoadTask(const HybridModel &model,
                                     const NodePtr &node,
                                     std::shared_ptr<NodeTask> &task) const {
  GE_CHECK_NOTNULL(node);
  std::string node_type = node->GetType();
  if (RefInputTask::IsBelong(node_type)) {
    GELOGI("node %s type %s is ref input task, use RefInputTask.",
           node->GetName().c_str(), node_type.c_str());
    task = MakeShared<RefInputTask>(node);
    if (task == nullptr) {
      REPORT_CALL_ERROR("E19999", "Create RefInputTask failed for node %s.", node->GetName().c_str());
      GELOGE(MEMALLOC_FAILED, "[Create][RefInputTask] failed for node %s.", node->GetName().c_str());
      return MEMALLOC_FAILED;
    }
  } else if (DependInputShapeTask::IsBelong(node_type)) {
    GELOGI("node %s type %s is depend input shape task, use DependInputShapeTask.",
           node->GetName().c_str(), node_type.c_str());
    task = MakeShared<DependInputShapeTask>(node);
    if (task == nullptr) {
      REPORT_CALL_ERROR("E19999", "Create DependInputShapeTask failed for node %s type %s.",
                        node->GetName().c_str(), node_type.c_str());
      GELOGE(MEMALLOC_FAILED, "[Create][DependInputShapeTask]failed for node %s type %s.",
             node->GetName().c_str(), node_type.c_str());
      return MEMALLOC_FAILED;
    }
  } else if (ConstantNodeTask::IsBelong(node_type)) {
    GELOGI("node %s type %s, use ConstantNodeTask.", node->GetName().c_str(), node_type.c_str());
    auto tensor = model.GetTensor(node);
    if (tensor == nullptr) {
      REPORT_CALL_ERROR("E19999", "GetTensor failed for name: %s", node->GetName().c_str());
      GELOGE(INTERNAL_ERROR, "[Get][Tensor] failed for name: %s", node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    task = MakeShared<ConstantNodeTask>(tensor);
    GE_CHECK_NOTNULL(task);
  } else if (NoOpNodeTask::IsBelong(node_type)) {
    GELOGI("node %s type %s , use NoOpNodeTask.", node->GetName().c_str(), node_type.c_str());
    task = MakeShared<NoOpNodeTask>();
    if (task == nullptr) {
      REPORT_CALL_ERROR("E19999", "Create NoOpNodeTask failed for NoOp node %s.", node->GetName().c_str());
      GELOGE(MEMALLOC_FAILED, "[Create][NoOpNodeTask]failed for NoOp node %s.", node->GetName().c_str());
      return MEMALLOC_FAILED;
    }
  } else {
    GELOGE(UNSUPPORTED, "node %s type %s is not support in GeLocalNodeExecutor now.",
        node->GetName().c_str(), node_type.c_str());
    return UNSUPPORTED;
  }
  return SUCCESS;
}

ConstantNodeTask::ConstantNodeTask(const TensorValue *tensor) : tensor_(tensor) {}

Status ConstantNodeTask::UpdateArgs(TaskContext &context) {
  return SUCCESS;
}

Status ConstantNodeTask::ExecuteAsync(TaskContext &context, std::function<void()> done_callback) {
  GELOGD("[%s] Start execute.", context.GetNodeName());
  GE_CHK_STATUS_RET(context.SetOutput(0, *tensor_), "[Set][Output] failed for [%s].", context.GetNodeName());
  if (done_callback) {
    GELOGD("[%s] Start invoke callback.", context.GetNodeName());
    done_callback();
  }

  GELOGD("[%s] Done execute successfully.", context.GetNodeName());
  return SUCCESS;
}

bool ConstantNodeTask::IsBelong(const std::string &op_type) {
  return constant_like_task_ops_.count(op_type) > 0;
}

Status NoOpNodeTask::UpdateArgs(TaskContext &context) {
  // no need to update args
  return SUCCESS;
}

Status NoOpNodeTask::ExecuteAsync(TaskContext &context, std::function<void()> done_callback) {
  GELOGD("[%s] Start execute.", context.GetNodeName());
  GE_CHK_STATUS_RET(context.TryExecuteCallback(done_callback));
  GELOGD("[%s] Done execute successfully.", context.GetNodeName());
  return SUCCESS;
}

bool NoOpNodeTask::IsBelong(const std::string &op_type) {
  return control_only_task_ops_.count(op_type) > 0;
}
}  // namespace hybrid
}  // namespace ge
