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

#include "hybrid/executor/worker/execution_engine.h"
#include "graph/runtime_inference_context.h"
#include "graph/load/model_manager/model_manager.h"
#include "hybrid/node_executor/node_executor.h"
#include "hybrid/executor//worker//shape_inference_engine.h"
#include "common/profiling/profiling_manager.h"

namespace ge {
namespace hybrid {
namespace {
constexpr int64_t kMaxPadding = 63;

Status LogInputs(const NodeItem &node_item, const TaskContext &task_context) {
  for (auto i = 0; i < task_context.NumInputs(); ++i) {
    const auto &input_tensor = task_context.GetInput(i);
    GE_CHECK_NOTNULL(input_tensor);
    const auto &tensor_desc = task_context.GetInputDesc(i);
    GE_CHECK_NOTNULL(tensor_desc);
    GELOGD("[%s] Print task args. input[%d] = %s, shape = [%s]",
           node_item.NodeName().c_str(),
           i,
           input_tensor->DebugString().c_str(),
           tensor_desc->GetShape().ToString().c_str());
  }

  return SUCCESS;
}

Status LogOutputs(const NodeItem &node_item, const TaskContext &task_context) {
  for (auto i = 0; i < task_context.NumOutputs(); ++i) {
    const auto &output_tensor = task_context.GetOutput(i);
    GE_CHECK_NOTNULL(output_tensor);
    const auto &tensor_desc = node_item.MutableOutputDesc(i);
    GE_CHECK_NOTNULL(tensor_desc);
    GELOGD("[%s] Print task args. output[%d] = %s, shape = [%s]",
           node_item.NodeName().c_str(),
           i,
           output_tensor->DebugString().c_str(),
           tensor_desc->MutableShape().ToString().c_str());
  }

  return SUCCESS;
}
}  // namespace

NodeDoneCallback::NodeDoneCallback(GraphExecutionContext *graph_context,
                                   std::shared_ptr<TaskContext> task_context)
    : graph_context_(graph_context), context_(std::move(task_context)) {
}

Status NodeDoneCallback::PrepareConstInputs(const NodeItem &node_item) {
  for (auto output_idx : node_item.to_const_output_id_list) {
    RECORD_CALLBACK_EVENT(graph_context_, node_item.NodeName().c_str(),
                          "[PrepareConstInputs] [index = %d] Start",
                          output_idx);

    auto output_tensor = context_->GetOutput(output_idx);
    GE_CHECK_NOTNULL(output_tensor);

    Tensor tensor;
    auto ge_tensor_desc = node_item.MutableOutputDesc(output_idx);
    GE_CHECK_NOTNULL(ge_tensor_desc);
    tensor.SetTensorDesc(TensorAdapter::GeTensorDesc2TensorDesc(*ge_tensor_desc));

    int64_t tensor_size;
    GE_CHK_GRAPH_STATUS_RET(TensorUtils::GetTensorSizeInBytes(*ge_tensor_desc, tensor_size),
                            "Failed to invoke GetTensorSizeInBytes");

    if (output_tensor->GetSize() < static_cast<size_t>(tensor_size)) {
      GELOGE(INTERNAL_ERROR,
          "[Check][Size][%s] Tensor size is not enough. output index = %d, required size = %ld, tensor = %s.",
          node_item.NodeName().c_str(), output_idx, tensor_size,
          output_tensor->DebugString().c_str());
      REPORT_INNER_ERROR("E19999",
                         "[%s] Tensor size is not enough. output index = %d, required size = %ld, tensor = %s.",
                         node_item.NodeName().c_str(), output_idx, tensor_size,
                         output_tensor->DebugString().c_str());
      return INTERNAL_ERROR;
    }

    vector<uint8_t> host_buffer(static_cast<unsigned long>(tensor_size));
    GELOGD("[%s] To cache output[%d] to host, size = %zu",
           node_item.NodeName().c_str(),
           output_idx,
           output_tensor->GetSize());
    if (tensor_size > 0) {
      GE_CHK_RT_RET(rtMemcpy(host_buffer.data(),
                             tensor_size,
                             output_tensor->GetData(),
                             tensor_size,
                             RT_MEMCPY_DEVICE_TO_HOST));
    }
    tensor.SetData(std::move(host_buffer));
    string context_id = std::to_string(graph_context_->context_id);
    RuntimeInferenceContext *runtime_infer_ctx = nullptr;
    GE_CHK_GRAPH_STATUS_RET(RuntimeInferenceContext::GetContext(context_id, &runtime_infer_ctx),
                            "Failed to get RuntimeInferenceContext, context_id = %s", context_id.c_str());
    GE_CHK_STATUS_RET(runtime_infer_ctx->SetTensor(node_item.node_id, output_idx, std::move(tensor)),
                      "[Set][Tensor] Failed, node = %s, output_index = %d", node_item.NodeName().c_str(), output_idx);
    GELOGD("[%s] Output[%d] cached successfully in context: %s. node_id = %d, shape = [%s]",
           node_item.NodeName().c_str(),
           output_idx,
           context_id.c_str(),
           node_item.node_id,
           ge_tensor_desc->GetShape().ToString().c_str());

    RECORD_CALLBACK_EVENT(graph_context_, node_item.NodeName().c_str(),
                          "[PrepareConstInputs] [index = %d] End",
                          output_idx);
  }

  return SUCCESS;
}

Status NodeDoneCallback::GetTaskDescInfo(const NodePtr node, const HybridModel *model,
                                         std::vector<TaskDescInfo> &task_desc_info) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(model);

  // only report aicpu and aicore node
  bool is_profiling_report = context_->GetNodeItem().is_profiling_report;
  if (!is_profiling_report) {
    GELOGD("Node[%s] is not aicore or aicpu, and no need to report data.", node->GetName().c_str());
    return SUCCESS;
  }

  GELOGD("GetTaskDescInfo of node [%s] start.", node->GetName().c_str());
  auto &prof_mgr = ProfilingManager::Instance();
  task_desc_info = context_->GetProfilingTaskDescInfo();
  context_->ClearProfilingTaskDescInfo();
  for (auto &tmp_task_desc : task_desc_info) {
    // save op input and output info
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    prof_mgr.GetOpInputOutputInfo(op_desc, tmp_task_desc);
  }

  return SUCCESS;
}

Status NodeDoneCallback::ProfilingReport() {
  auto node = context_->GetNodeItem().node;
  if (node == nullptr) {
    GELOGE(PARAM_INVALID, "[Get][Node] value is nullptr.");
    REPORT_INNER_ERROR("E19999", "TaskContext GetNodeItem value is nullptr.");
    return PARAM_INVALID;
  }

  const auto &op_type = node->GetType();
  if (op_type == PARTITIONEDCALL) {
    return SUCCESS;
  }

  GE_CHECK_NOTNULL(graph_context_);
  const HybridModel *model = graph_context_->model;
  GE_CHECK_NOTNULL(model);

  GELOGD("ProfilingReport of node [%s] model [%s] start.", node->GetName().c_str(), model->GetModelName().c_str());
  std::vector<TaskDescInfo> task_desc_info;
  auto profiling_ret = GetTaskDescInfo(node, model, task_desc_info);
  if (profiling_ret != RT_ERROR_NONE) {
    GELOGE(profiling_ret, "[Get][TaskDescInfo] of node:%s failed.", node->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "GetTaskDescInfo of node:%s failed.", node->GetName().c_str());
    return profiling_ret;
  }

  auto &profiling_manager = ProfilingManager::Instance();
  profiling_manager.ReportProfilingData(model->GetModelId(), task_desc_info);
  return SUCCESS;
}

Status NodeDoneCallback::DumpDynamicNode() {
  auto node = context_->GetNodeItem().node;
  if (node == nullptr) {
    GELOGE(PARAM_INVALID, "[Get][Node] value is nullptr.");
    REPORT_INNER_ERROR("E19999", "get node value is nullptr.");
    return PARAM_INVALID;
  }
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(graph_context_);
  const HybridModel *model = graph_context_->model;
  GE_CHECK_NOTNULL(model);
  std::string dynamic_model_name = model->GetModelName();
  std::string dynamic_om_name = model->GetOmName();
  uint32_t model_id = model->GetModelId();
  if (!context_->GetDumpProperties().IsLayerNeedDump(dynamic_model_name, dynamic_om_name, op_desc->GetName())) {
    GELOGI("[%s] is not in dump list, no need dump", op_desc->GetName().c_str());
    return SUCCESS;
  }
  dump_op_.SetDynamicModelInfo(dynamic_model_name, dynamic_om_name, model_id);

  auto stream = context_->GetStream();
  vector<uintptr_t> input_addrs;
  vector<uintptr_t> output_addrs;
  for (int i = 0; i < context_->NumInputs(); i++) {
    auto tensor_value = context_->GetInput(i);
    GE_CHK_BOOL_RET_STATUS(tensor_value != nullptr, PARAM_INVALID, "[Get][Tensor] value is nullptr.");
    uintptr_t input_addr = reinterpret_cast<uintptr_t>(tensor_value->GetData());
    input_addrs.emplace_back(input_addr);
  }
  for (int j = 0; j < context_->NumOutputs(); j++) {
    auto tensor_value = context_->GetOutput(j);
    GE_CHK_BOOL_RET_STATUS(tensor_value != nullptr, PARAM_INVALID, "[Get][Tensor] value is nullptr.");
    uintptr_t output_addr = reinterpret_cast<uintptr_t>(tensor_value->GetData());
    output_addrs.emplace_back(output_addr);
  }

  dump_op_.SetDumpInfo(context_->GetDumpProperties(), op_desc, input_addrs, output_addrs, stream);

  void *loop_per_iter = nullptr;
  TensorValue *varible_loop_per_iter = context_->GetVariable(NODE_NAME_FLOWCTRL_LOOP_PER_ITER);
  if (varible_loop_per_iter != nullptr) {
    loop_per_iter = const_cast<void *>(varible_loop_per_iter->GetData());
  }

  void *loop_cond = nullptr;
  TensorValue *varible_loop_cond = context_->GetVariable(NODE_NAME_FLOWCTRL_LOOP_COND);
  if (varible_loop_cond != nullptr) {
    loop_cond = const_cast<void *>(varible_loop_cond->GetData());
  }
  void *global_step = context_->GetExecutionContext()->global_step;
  dump_op_.SetLoopAddr(global_step, loop_per_iter, loop_cond);

  GE_CHK_STATUS_RET(dump_op_.LaunchDumpOp(), "[Launch][DumpOp] failed in hybird model.");

  auto rt_ret = rtStreamSynchronize(stream);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(rt_ret, "[Call][rtStreamSynchronize] failed, ret = %d.", rt_ret);
    REPORT_CALL_ERROR("E19999", "call rtStreamSynchronize failed, ret = %d.", rt_ret);
    return rt_ret;
  }
  return SUCCESS;
}

Status NodeDoneCallback::SaveDumpOpInfo() {
  GE_CHECK_NOTNULL(graph_context_);
  GE_CHECK_NOTNULL(graph_context_->model);

  auto node = context_->GetNodeItem().node;
  if (node == nullptr) {
    GELOGE(PARAM_INVALID, "[Save][DumpOpInfo] Get node is nullptr.");
    return PARAM_INVALID;
  }
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  vector<void *> input_addrs;
  vector<void *> output_addrs;
  for (int i = 0; i < context_->NumInputs(); i++) {
    auto tensor_value = context_->GetInput(i);
    GE_CHK_BOOL_RET_STATUS(tensor_value != nullptr, PARAM_INVALID, "[Save][DumpOpInfo] Tensor value is nullptr.");
    void *input_addr = const_cast<void *>(tensor_value->GetData());
    input_addrs.emplace_back(input_addr);
  }
  for (int j = 0; j < context_->NumOutputs(); j++) {
    auto tensor_value = context_->GetOutput(j);
    GE_CHK_BOOL_RET_STATUS(tensor_value != nullptr, PARAM_INVALID, "[Save][DumpOpInfo] Tensor value is nullptr.");
    void *output_addr = const_cast<void *>(tensor_value->GetData());
    output_addrs.emplace_back(output_addr);
  }

  uint32_t stream_id = context_->GetStreamId();
  uint32_t task_id = context_->GetTaskId();
  graph_context_->exception_dumper.SaveDumpOpInfo(op_desc, task_id, stream_id, input_addrs, output_addrs);

  return SUCCESS;
}

Status NodeDoneCallback::OnNodeDone() {
  auto &node_item = context_->GetNodeItem();
  GELOGI("[%s] Start callback process.", node_item.NodeName().c_str());
  RECORD_CALLBACK_EVENT(graph_context_, context_->GetNodeName(), "[Compute] End");
  RECORD_CALLBACK_EVENT(graph_context_, context_->GetNodeName(), "[Callback] Start");

  const DumpProperties &dump_properties = context_->GetDumpProperties();
  if (dump_properties.IsDumpOpen() || context_->IsOverFlow()) {
    GELOGI("Start to dump dynamic shape op");
    GE_CHK_STATUS_RET(DumpDynamicNode(), "[Call][DumpDynamicNode] Failed.");
  }

  auto model_manager = ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  if (model_manager->IsDumpExceptionOpen()) {
    GE_CHK_STATUS_RET(SaveDumpOpInfo(), "[Save][DumpOpInfo] Failed to dump op info.");
  }

  if (ProfilingManager::Instance().ProfilingModelLoadOn()) {
    GE_CHK_STATUS_RET(ProfilingReport(), "[Report][Profiling] of node[%s] failed.", node_item.NodeName().c_str());
  }

  // release workspace
  context_->ReleaseWorkspace();
  // release inputs
  for (int i = 0; i < context_->NumInputs(); ++i) {
    context_->ReleaseInput(i);
  }

  GE_CHK_STATUS_RET_NOLOG(PrepareConstInputs(node_item));
  if (node_item.shape_inference_type == DEPEND_SHAPE_RANGE || node_item.shape_inference_type == DEPEND_COMPUTE) {
    // update output tensor sizes
    const auto &guard = node_item.MutexGuard("OnNodeDone");
    GE_CHK_STATUS_RET_NOLOG(ShapeInferenceEngine::CalcOutputTensorSizes(node_item));
    GE_CHK_STATUS_RET_NOLOG(context_->GetNodeState()->GetShapeInferenceState().UpdateOutputDesc());
    (void)guard;
  }
  // PropagateOutputs for type == DEPEND_COMPUTE
  if (node_item.shape_inference_type == DEPEND_COMPUTE) {
    if (graph_context_->trace_enabled) {
      (void) LogOutputs(node_item, *context_);
    }

    GE_CHK_STATUS_RET(context_->PropagateOutputs(), "[Propagate][Outputs] of [%s] failed.",
                      node_item.NodeName().c_str());

    RECORD_CALLBACK_EVENT(graph_context_, context_->GetNodeName(), "[PropagateOutputs] End");
  }

  // release condition variable
  if (node_item.has_observer) {
    GELOGI("[%s] Notify observer. node_id = %d", node_item.NodeName().c_str(), node_item.node_id);
    context_->NodeDone();
  }

  RECORD_CALLBACK_EVENT(graph_context_, context_->GetNodeName(), "[Callback] End");
  return SUCCESS;
}

Status ExecutionEngine::ExecuteAsync(NodeState &node_state,
                                     const std::shared_ptr<TaskContext> &task_context,
                                     GraphExecutionContext &execution_context,
                                     const std::function<void()> &callback) {
  GELOGI("[%s] Node is ready for execution", task_context->GetNodeName());
  RECORD_EXECUTION_EVENT(&execution_context, task_context->GetNodeName(), "Start");
  GE_CHK_STATUS_RET_NOLOG(DoExecuteAsync(node_state, *task_context, execution_context, callback));
  GE_CHK_STATUS_RET_NOLOG(PropagateOutputs(*node_state.GetNodeItem(), *task_context, execution_context));
  return SUCCESS;
}

Status ExecutionEngine::DoExecuteAsync(NodeState &node_state,
                                       TaskContext &task_context,
                                       GraphExecutionContext &context,
                                       const std::function<void()> &callback) {
  const auto &task = node_state.GetKernelTask();
  if (task == nullptr) {
    GELOGE(INTERNAL_ERROR, "[Get][KernelTask] of [%s] is null.", node_state.GetName().c_str());
    REPORT_INNER_ERROR("E19999", "GetKernelTask of %s failed.", node_state.GetName().c_str());
    return INTERNAL_ERROR;
  }

  // Wait for dependent nodes(DEPEND_COMPUTE), so that the input tensors are valid.
  RECORD_EXECUTION_EVENT(&context, task_context.GetNodeName(), "[AwaitDependents] Start");
  HYBRID_CHK_STATUS_RET(node_state.AwaitInputTensors(context),
                        "[%s] Failed to wait for dependent nodes.",
                        node_state.GetName().c_str());

  const auto &node_item = *node_state.GetNodeItem();
  auto executor = node_item.node_executor;
  GE_CHECK_NOTNULL(executor);
  RECORD_EXECUTION_EVENT(&context, task_context.GetNodeName(), "[PrepareTask] Start");
  node_state.UpdatePersistTensor();
  GE_CHK_STATUS_RET(executor->PrepareTask(*task, task_context), "[Prepare][Task] for [%s] failed.",
                    node_state.GetName().c_str());
  RECORD_EXECUTION_EVENT(&context, task_context.GetNodeName(), "[PrepareTask] End");
  GELOGD("[%s] Done task preparation successfully.", node_state.GetName().c_str());

  if (context.trace_enabled) {
    LogInputs(node_item, task_context);
    if (node_item.shape_inference_type != DEPEND_COMPUTE) {
      LogOutputs(node_item, task_context);
    }
  }

  GE_CHK_STATUS_RET(ValidateInputTensors(node_state, task_context), "[Validate][InputTensors] for %s failed.",
                    node_state.GetName().c_str());
  RECORD_EXECUTION_EVENT(&context, task_context.GetNodeName(), "[ValidateInputTensors] End");

  if (GraphExecutionContext::profiling_level > 0) {
    auto *ctx = &context;
    const string &name = node_state.GetName();
    (void)task_context.RegisterCallback([ctx, name]() {
      RECORD_CALLBACK_EVENT(ctx, name.c_str(), "[Compute] Start");
    });
  }
  RECORD_EXECUTION_EVENT(&context, task_context.GetNodeName(), "[ExecuteTask] Start");
  HYBRID_CHK_STATUS_RET(node_item.node_executor->ExecuteTask(*task, task_context, callback),
                        "[%s] Failed to execute task",
                        node_state.GetName().c_str());
  RECORD_EXECUTION_EVENT(&context, task_context.GetNodeName(), "[ExecuteTask] End");

  GELOGD("[%s] Done task launch successfully.", node_state.GetName().c_str());
  return SUCCESS;
}

Status ExecutionEngine::ValidateInputTensors(const NodeState &node_state, const TaskContext &task_context) {
  for (auto i = 0; i < task_context.NumInputs(); ++i) {
    const auto &input_tensor = task_context.GetInput(i);
    GE_CHECK_NOTNULL(input_tensor);
    if (input_tensor->GetData() == nullptr) {
      GELOGD("[%s] Skipping null input, index = %d", task_context.GetNodeName(), i);
      continue;
    }

    const auto &tensor_desc = task_context.MutableInputDesc(i);
    GE_CHECK_NOTNULL(tensor_desc);
    if (tensor_desc->GetDataType() == DT_STRING) {
      GELOGD("[%s] Skipping DT_STRING input, index = %d", task_context.GetNodeName(), i);
      continue;
    }

    if (input_tensor->GetData() == nullptr) {
      GELOGD("[%s] Skipping null input, index = %d", task_context.GetNodeName(), i);
      continue;
    }

    int64_t expected_size = 0;
    (void)TensorUtils::GetSize(*tensor_desc, expected_size);
    GELOGD("[%s] Input[%d] expects [%ld] bytes.", task_context.GetNodeName(), i, expected_size);
    auto size_diff = expected_size - static_cast<int64_t>(input_tensor->GetSize());
    if (size_diff > 0) {
      if (size_diff <= kMaxPadding) {
        GELOGW("[%s] Input[%d]: tensor size mismatches. expected: %ld, but given %zu",
               task_context.GetNodeName(),
               i,
               expected_size,
               input_tensor->GetSize());
      } else {
        GELOGE(INTERNAL_ERROR,
               "[Check][Size] for [%s] Input[%d]: tensor size mismatches. expected: %ld, but given %zu.",
               task_context.GetNodeName(), i, expected_size, input_tensor->GetSize());
        REPORT_INNER_ERROR("E19999", "[%s] Input[%d]: tensor size mismatches. expected: %ld, but given %zu.",
                           task_context.GetNodeName(), i, expected_size, input_tensor->GetSize());
        return INTERNAL_ERROR;
      }
    }
  }

  return SUCCESS;
}

Status ExecutionEngine::PropagateOutputs(const NodeItem &node_item,
                                         TaskContext &task_context,
                                         GraphExecutionContext &context) {
  if (node_item.shape_inference_type != DEPEND_COMPUTE) {
    GE_CHK_STATUS_RET(task_context.PropagateOutputs(), "[Propagate][Outputs] for [%s] failed.",
                      node_item.NodeName().c_str());
    RECORD_EXECUTION_EVENT(&context, task_context.GetNodeName(), "[PropagateOutputs] End");
    GELOGD("[%s] Done propagating outputs successfully.", node_item.NodeName().c_str());
  }

  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
