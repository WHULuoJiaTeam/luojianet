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

#include "hybrid/executor/hybrid_model_executor.h"
#include "graph/ge_context.h"
#include "graph/runtime_inference_context.h"
#include "graph/utils/tensor_utils.h"
#include "graph/load/model_manager/model_manager.h"
#include "common/dump/dump_manager.h"
#include "common/profiling/profiling_manager.h"

namespace ge {
namespace hybrid {
namespace {
const int kIntBase = 10;
const char *const kEnvProfilingLevel = "HYBRID_PROFILING_LEVEL";
} // namespace
HybridModelExecutor::HybridModelExecutor(HybridModel *model, uint32_t device_id, rtStream_t stream)
    : model_(model), device_id_(device_id), stream_(stream) {
}

HybridModelExecutor::~HybridModelExecutor() {
}

Status HybridModelExecutor::Init(ThreadPool *thread_pool) {
  GELOGD("Start to init HybridGraphEngine.");
  GE_CHK_STATUS_RET_NOLOG(InitExecutionContext());
  root_graph_executor_.reset(
    new (std::nothrow) SubgraphExecutor(model_->GetRootGraphItem(), &context_, false, thread_pool));
  GE_CHECK_NOTNULL(root_graph_executor_);
  GELOGD("HybridGraphEngine initialized successfully.");
  return SUCCESS;
}

Status HybridModelExecutor::Execute(HybridModelExecutor::ExecuteArgs &args) {
  GELOGD("Start to execute model.");
  auto root_graph_item = model_->GetRootGraphItem();
  GE_CHECK_NOTNULL(root_graph_item);

  if (root_graph_item->IsDynamic() && !model_->IsSingleOp()) {
    GE_CHK_STATUS_RET(CheckInputShapeByShapeRange(root_graph_item, args),
                      "[%s] check input node shape by shape range failed.",
                      root_graph_item->GetName().c_str());
  }

  if (context_.global_step != nullptr) {
    GE_CHK_RT_RET(rtMemcpyAsync(context_.global_step, sizeof(uint64_t), &context_.iteration,
                                sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE_EX, context_.stream));
  }
  auto ret = ExecuteGraphInternal(args);
  Cleanup();
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[Cleanup] End");
  GELOGD("Model executed successfully.");
  if (context_.profiler != nullptr) {
    context_.profiler->Dump(std::cout);
    context_.profiler->Reset();
  }
  root_graph_executor_->ReleaseContext();

  context_.iteration += 1;
  if (ret == END_OF_SEQUENCE) {
    args.is_eos = true;
  } else {
    GE_CHK_STATUS_RET(ret, "[Invoke][ExecuteGraphInternal] Failed, ret:%d.", ret);
  }
  return SUCCESS;
}

Status HybridModelExecutor::ExecuteGraphInternal(HybridModelExecutor::ExecuteArgs &args) {
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[InitContext] Start");
  GE_CHK_STATUS_RET_NOLOG(ResetExecutionContext(context_));
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[InitContext] End");

  uint64_t index_id = context_.iteration + 1;
  uint64_t model_id = static_cast<uint64_t>(model_->GetModelId());
  int32_t device_id = static_cast<int32_t>(device_id_);
  auto &prof_mgr = ProfilingManager::Instance();
  // tag_id 0 means step begin, 1 meas step end.
  if (!model_->IsSingleOp()) {
    GE_CHK_STATUS_RET_NOLOG(prof_mgr.ProfileStepInfo(index_id, model_id, 0, stream_, device_id));
  }

  HYBRID_CHK_STATUS_RET(root_graph_executor_->ExecuteAsync(args.inputs, args.input_desc, args.outputs),
                        "Failed to execute partitioned call.");
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[ExecuteAsync] End");

  if (!model_->IsSingleOp()) {
    GE_CHK_STATUS_RET_NOLOG(prof_mgr.ProfileStepInfo(index_id, model_id, 1, stream_, device_id));
  }

  if (!model_->IsSingleOp()) {
    Status ret = root_graph_executor_->Synchronize();
    if (ret != ge::SUCCESS) {
      auto model_manager = ModelManager::GetInstance();
      GE_CHECK_NOTNULL(model_manager);
      auto exception_infos = model_manager->GetExceptionInfos();
      if (!exception_infos.empty()) {
        HYBRID_CHK_STATUS_RET(context_.DumpExceptionInfo(exception_infos),
                              "[Execute][GraphInternal] Dump exception info failed.");
      }
      if (ret == ge::END_OF_SEQUENCE) {
        GELOGD("Got end of sequence");
      } else {
        GELOGE(ret, "[Execute][GraphInternal] Synchronize failed.");
      }
      return ret;
    }
    RECORD_MODEL_EXECUTION_EVENT(&context_, "[Synchronize] End");
  }

  args.outputs.clear();
  HYBRID_CHK_STATUS_RET(root_graph_executor_->GetOutputs(args.outputs, args.output_desc), "Failed to get outputs");
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[GetOutput] End");
  return SUCCESS;
}

Status HybridModelExecutor::Cleanup() {
  GELOGD("Start to cleanup.");
  context_.callback_manager->Destroy();
  RuntimeInferenceContext::DestroyContext(std::to_string(context_.context_id));
  GELOGD("Cleanup successfully.");
  return SUCCESS;
}

Status HybridModelExecutor::InitExecutionContext() {
  GE_CHK_RT_RET(rtCtxGetCurrent(&context_.rt_context));
  GE_CHK_RT_RET(rtCtxSetCurrent(context_.rt_context));

  context_.global_step = model_->GetGlobalStep();
  context_.stream = stream_;
  context_.model = model_;
  context_.is_eos_ = false;
  context_.session_id = ::ge::GetContext().SessionId();
  context_.ge_context = &GetThreadLocalContext();
  GELOGD("session id from model = %lu, from context = %lu", model_->GetSessionId(), context_.session_id);
  context_.allocator = NpuMemoryAllocator::GetAllocator(device_id_);
  GE_CHECK_NOTNULL(context_.allocator);
  context_.callback_manager = std::unique_ptr<CallbackManager>(new(std::nothrow)CallbackManager());
  GE_CHECK_NOTNULL(context_.callback_manager);
  context_.dump_properties = DumpManager::GetInstance().GetDumpProperties(context_.session_id);
  const char *profiling_level = std::getenv(kEnvProfilingLevel);
  if (profiling_level != nullptr) {
    GraphExecutionContext::profiling_level = std::strtol(profiling_level, nullptr, kIntBase);
    GELOGD("Got profiling level = %ld", GraphExecutionContext::profiling_level);
    if (GraphExecutionContext::profiling_level > 0) {
      context_.profiler.reset(new(std::nothrow)HybridProfiler());
      GE_CHECK_NOTNULL(context_.profiler);
    }
  }

  if (IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    context_.trace_enabled = true;
  }
  return SUCCESS;
}

Status HybridModelExecutor::ResetExecutionContext(GraphExecutionContext &context) {
  GE_CHK_STATUS_RET_NOLOG(context.callback_manager->Init());
  string ctx_id = std::to_string(context.context_id);
  RuntimeInferenceContext::DestroyContext(ctx_id);
  GE_CHK_GRAPH_STATUS_RET(RuntimeInferenceContext::CreateContext(ctx_id), "Failed to Destroy RuntimeInferenceContext");
  RuntimeInferenceContext *ctx = nullptr;
  GE_CHK_GRAPH_STATUS_RET(RuntimeInferenceContext::GetContext(ctx_id, &ctx), "Failed to get context");
  for (auto &host_tensor : context.model->GetHostTensors()) {
    auto node_id = host_tensor.first;
    for (const auto &output_idx_and_tensor : host_tensor.second) {
      auto output_idx = output_idx_and_tensor.first;
      GELOGD("Preload const host tensor, node_id = %ld, output id = %d", node_id, output_idx);
      ctx->SetTensor(node_id, output_idx, output_idx_and_tensor.second.Clone());
    }
  }
  return SUCCESS;
}

Status HybridModelExecutor::CheckInputShapeByShapeRange(const GraphItem *graph_item,
                                                        HybridModelExecutor::ExecuteArgs &args) {
  GE_CHECK_NOTNULL(graph_item);
  auto input_nodes = graph_item->GetInputNodes();
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    auto &input_node = input_nodes[i];
    if (input_node == nullptr) {
      GELOGD("[%s] Input[%zu] is not needed by graph, skip it.", graph_item->GetName().c_str(), i);
      continue;
    }
    if (!input_node->is_dynamic) {
      GELOGD("[%s] Input[%zu] is not dynamic, skip it.", graph_item->GetName().c_str(), i);
      continue;
    }
    GeTensorDescPtr model_input_desc = input_node->MutableInputDesc(0);
    GE_CHECK_NOTNULL(model_input_desc);
    std::vector<std::pair<int64_t, int64_t>> shape_range;
    if (model_input_desc->GetShapeRange(shape_range) != SUCCESS) {
      REPORT_INNER_ERROR("E19999", "[%s] Input[%zu] get shape range failed", graph_item->GetName().c_str(), i);
      GELOGE(INTERNAL_ERROR, "[%s] Input[%zu] get shape range failed", graph_item->GetName().c_str(), i);
      return INTERNAL_ERROR;
    }
    if (shape_range.empty()) {
      GELOGD("[%s] Input[%zu] shape is not needed to check by shape range, skip it.", graph_item->GetName().c_str(), i);
      continue;
    }
    if (i >= args.input_desc.size()) {
      REPORT_INNER_ERROR("E19999", "[%s] Inputs[%zu] is greater than or equal to input desc size[%zu].",
                         graph_item->GetName().c_str(), i, args.input_desc.size());
      GELOGE(INTERNAL_ERROR, "[%s] inputs[%zu] is greater than or equal to input desc size[%zu].",
             graph_item->GetName().c_str(), i, args.input_desc.size());
      return INTERNAL_ERROR;
    }
    ConstGeTensorDescPtr args_tensor_desc = args.input_desc[i];
    GE_CHECK_NOTNULL(args_tensor_desc);
    GeShape shape = args_tensor_desc->GetShape();
    if (shape.IsUnknownShape()) {
      REPORT_INNER_ERROR("E19999", "[%s] Input desc shape [%zu] designed by user must be static.",
                         graph_item->GetName().c_str(), i);
      GELOGE(INTERNAL_ERROR, "[%s] Input desc shape [%zu] designed by user must be static.",
             graph_item->GetName().c_str(), i);
      return INTERNAL_ERROR;
    }

    if (TensorUtils::CheckShapeByShapeRange(shape, shape_range) != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Check][InputShape] [%s] check input [%zu] shape failed by shape range.",
             graph_item->GetName().c_str(), i);
      return PARAM_INVALID;
    }
  }

  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
