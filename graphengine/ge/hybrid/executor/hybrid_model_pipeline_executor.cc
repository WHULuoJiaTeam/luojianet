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

#include "hybrid/executor/hybrid_model_pipeline_executor.h"

#include "common/math/math_util.h"
#include "common/dump/dump_manager.h"
#include "graph/ge_context.h"
#include "graph/runtime_inference_context.h"
#include "graph/load/model_manager/model_manager.h"

namespace ge {
namespace hybrid {
namespace {
constexpr int kNumExecutors = 2;
const int kMinLoopCount = 2;
const int kIntBase = 10;
const char *const kEnvProfilingLevel = "HYBRID_PROFILING_LEVEL";
}

StageExecutor::StageExecutor(int id, HybridModel *model, PipeExecutionConfig *config)
    : id_(id), model_(model), pipe_config_(config) {}

StageExecutor::~StageExecutor() {
  GELOGD("~StageExecutor(), id = %d", id_);
  if (stream_ != nullptr) {
    GE_CHK_RT(rtStreamDestroy(stream_));
    stream_ = nullptr;
  }
  if (hccl_stream_ != nullptr) {
    GE_CHK_RT(rtStreamDestroy(hccl_stream_));
    hccl_stream_ = nullptr;
  }
}

Status StageExecutor::Init() {
  GELOGD("[Executor: %d] Start to init StateExecutor", id_);
  context_.rt_context = pipe_config_->rt_context;
  GE_CHK_STATUS_RET_NOLOG(InitExecutionContext());
  GE_CHK_RT_RET(rtStreamCreate(&stream_, RT_STREAM_PRIORITY_DEFAULT));
  GE_CHK_RT_RET(rtStreamCreate(&hccl_stream_, RT_STREAM_PRIORITY_DEFAULT));
  context_.stream = stream_;
  context_.hccl_stream = hccl_stream_;

  root_graph_executor_.reset(new (std::nothrow) SubgraphExecutor(model_->GetRootGraphItem(), &context_));
  GE_CHECK_NOTNULL(root_graph_executor_);

  GELOGD("[Executor: %d] Init stage executor successfully", id_);
  return SUCCESS;
}

Status StageExecutor::ResetExecutionContext(GraphExecutionContext &context) {
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

Status StageExecutor::Start(const std::vector<TensorValue> &inputs, const std::vector<ConstGeTensorDescPtr> &input_desc,
                            int iteration_count) {
  GELOGD("Start");
  GE_CHK_RT_RET(rtCtxSetCurrent(context_.rt_context));
  int num_loops = iteration_count / pipe_config_->num_executors;
  if (id_ < iteration_count % iteration_count) {
    num_loops += 1;
  }
  FMK_INT32_MULCHECK(num_loops, pipe_config_->num_stages);
  num_loops *= pipe_config_->num_stages;
  GELOGD("[Executor: %d] loop count = %d", id_, num_loops);

  for (int loop_idx = 0; loop_idx < num_loops; ++loop_idx) {
    GELOGD("[Executor: %d] Start to wait for task.", id_);
    StageTask task_info;
    task_queue_.Pop(task_info);
    GELOGD("[Executor: %d] Got task, stage = %d, iteration = %ld", id_, task_info.stage, task_info.iteration);
    if (task_info.iteration >= pipe_config_->iteration_end) {
      GELOGE(INTERNAL_ERROR, "[Check][Range][Executor: %d] Unexpected iteration: %ld.", id_, task_info.iteration);
      REPORT_INNER_ERROR("E19999", "[Executor: %d] Unexpected iteration: %ld.", id_, task_info.iteration);
      return INTERNAL_ERROR;
    }

    if (task_info.event != nullptr) {
      GELOGD("[%d] Add StreamWaitEvent", id_);
      GE_CHK_RT_RET(rtStreamWaitEvent(stream_, task_info.event));
      RECORD_MODEL_EXECUTION_EVENT(&context_, "[iteration = %ld] [Stage = %d] EventWait End", task_info.iteration,
                                   task_info.stage);
    }

    RECORD_MODEL_EXECUTION_EVENT(&context_, "[iteration = %ld] [Stage = %d] Start", task_info.iteration,
                                 task_info.stage);

    if (task_info.stage == 0) {
      GELOGD("[Executor: %d] To ResetExecutionContext", id_);
      GE_CHK_STATUS_RET(ResetExecutionContext(context_),
                        "[Invoke][ResetExecutionContext][Executor: %d] Failed to reset context", id_);
      context_.iteration = task_info.iteration;
      GE_CHK_STATUS_RET_NOLOG(SetInputs(inputs, input_desc));
    }

    RECORD_MODEL_EXECUTION_EVENT(&context_, "[Stage = %d] PartialExecuteAsync Start", task_info.stage);
    GE_CHK_STATUS_RET(root_graph_executor_->PartialExecuteAsync(task_info.stage));
    RECORD_MODEL_EXECUTION_EVENT(&context_, "[Stage = %d] PartialExecuteAsync End", task_info.stage);
    GELOGD("[Executor: %d] PartialExecuteAsync successfully.", id_);

    // notify next execution unit
    StageTask next_task;
    next_task.stage = task_info.stage;
    next_task.iteration = task_info.iteration + 1;
    if ((task_info.iteration + 1) % iteration_count > 0) {
      GE_CHK_RT_RET(rtEventCreate(&next_task.event));
      GE_CHK_RT_RET(rtEventRecord(next_task.event, context_.hccl_stream));
    }

    auto sync_result = Synchronize();
    if (sync_result != SUCCESS) {
      GELOGE(sync_result,
             "[Invoke][Synchronize][Executor: %d] Failed to sync result:%d. iteration = %ld",
             id_, sync_result, task_info.iteration);
      REPORT_CALL_ERROR("E19999", "[Executor: %d] Failed to sync result:%d. iteration = %ld",
                        id_, sync_result, task_info.iteration);
      if (context_.profiler != nullptr) {
        context_.profiler->Dump(std::cout);
      }
      context_.callback_manager->Destroy();
      RuntimeInferenceContext::DestroyContext(std::to_string(context_.context_id));
      return sync_result;
    }
    if (task_info.event != nullptr) {
      GE_CHK_RT_RET(rtEventDestroy(task_info.event));
      RECORD_MODEL_EXECUTION_EVENT(&context_, "[iteration = %ld] [Stage = %d] EventDestroy End", task_info.iteration,
                                   task_info.stage);
    }

    RECORD_MODEL_EXECUTION_EVENT(&context_, "[iteration = %ld] [Stage = %d] End", task_info.iteration, task_info.stage);

    // if end stage
    if (task_info.stage >= pipe_config_->num_stages - 1) {
      RECORD_MODEL_EXECUTION_EVENT(&context_, "[iteration = %ld] Schedule End", task_info.iteration);
      GELOGD("[Executor: %d] End of iteration [%ld]", id_, task_info.iteration);
      context_.callback_manager->Destroy();
      RuntimeInferenceContext::DestroyContext(std::to_string(context_.context_id));
    }
    next_executor_->ExecuteAsync(next_task);
    GELOGD("[Executor: %d] Push item successfully.", id_);
  }

  GELOGD("[Executor: %d] Process task ended.", id_);
  return SUCCESS;
}

Status StageExecutor::ExecuteAsync(const StageTask &args) {
  (void)task_queue_.Push(args);
  return SUCCESS;
}

Status StageExecutor::Synchronize() {
  auto ret = root_graph_executor_->Synchronize();
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[Synchronize] End, ret = %u", ret);
  return ret;
}

HybridModelPipelineExecutor::HybridModelPipelineExecutor(HybridModel *model, uint32_t device_id)
    : model_(model), device_id_(device_id) {
  config_.num_executors = kNumExecutors;
  config_.num_stages = model_->GetRootGraphItem()->NumGroups();
  config_.device_id = device_id_;
  config_.iteration_end = 0;
}

Status StageExecutor::InitExecutionContext() {
  GE_CHK_RT_RET(rtCtxSetCurrent(context_.rt_context));

  context_.model = model_;
  context_.session_id = ::ge::GetContext().SessionId();
  GELOGD("session id from model = %lu, from context = %lu", model_->GetSessionId(), context_.session_id);
  context_.allocator = NpuMemoryAllocator::GetAllocator(pipe_config_->device_id);
  GE_CHECK_NOTNULL(context_.allocator);
  context_.callback_manager = std::unique_ptr<CallbackManager>(new (std::nothrow) CallbackManager());
  GE_CHECK_NOTNULL(context_.callback_manager);
  context_.dump_properties = DumpManager::GetInstance().GetDumpProperties(context_.session_id);
  context_.is_eos_ = false;
  if (IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    context_.trace_enabled = true;
  }
  return SUCCESS;
}

Status StageExecutor::SetInputs(const vector<TensorValue> &inputs, const vector<ConstGeTensorDescPtr> &input_desc) {
  root_graph_executor_->InitForPartialExecution(inputs, input_desc);
  return SUCCESS;
}

Status StageExecutor::GetOutputs(vector<TensorValue> &outputs, vector<ConstGeTensorDescPtr> &output_desc) {
  return root_graph_executor_->GetOutputs(outputs, output_desc);
}

void StageExecutor::Reset() {
  task_queue_.Stop();
  task_queue_.Clear();
  task_queue_.Restart();
}

Status HybridModelPipelineExecutor::Init() {
  const char *profiling_level = std::getenv(kEnvProfilingLevel);
  if (profiling_level != nullptr) {
    GraphExecutionContext::profiling_level = std::strtol(profiling_level, nullptr, kIntBase);
    GELOGD("Got profiling level = %ld", GraphExecutionContext::profiling_level);
    if (GraphExecutionContext::profiling_level > 0) {
      context_.profiler.reset(new (std::nothrow) HybridProfiler());
      GE_CHECK_NOTNULL(context_.profiler);
    }
  }

  GELOGD("Number of stages = %d, number of executors = %d", config_.num_stages, config_.num_executors);
  GE_CHK_RT_RET(rtCtxGetCurrent(&config_.rt_context));
  GE_CHK_STATUS_RET_NOLOG(InitStageExecutors());
  return SUCCESS;
}

Status HybridModelPipelineExecutor::InitStageExecutors() {
  for (int i = 0; i < config_.num_executors; ++i) {
    auto stage_executor = std::unique_ptr<StageExecutor>(new (std::nothrow) StageExecutor(i, model_, &config_));
    GE_CHECK_NOTNULL(stage_executor);
    GE_CHK_STATUS_RET_NOLOG(stage_executor->Init());

    if (context_.profiler != nullptr) {
      // will call unique_ptr::release later
      stage_executor->context_.profiler.reset(context_.profiler.get());
    }

    stage_executors_.emplace_back(std::move(stage_executor));
  }

  // build propagation loop
  for (int i = 0; i < config_.num_executors - 1; ++i) {
    stage_executors_[i]->SetNext(stage_executors_[i + 1].get());
  }
  stage_executors_[config_.num_executors - 1]->SetNext(stage_executors_[0].get());
  return SUCCESS;
}

Status HybridModelPipelineExecutor::Execute(HybridModelExecutor::ExecuteArgs &args) {
  int loop_count = args.num_loops;
  GE_CHECK_GE(loop_count, kMinLoopCount);

  auto &inputs = args.inputs;
  auto &input_desc = args.input_desc;
  // Start schedulers
  std::vector<std::future<Status>> futures;
  for (size_t i = 0; i < stage_executors_.size(); ++i) {
    GELOGD("Starting executor %zu", i);
    auto executor = stage_executors_[i].get();
    executor->Reset();
    auto future = std::async(
        [loop_count, executor, inputs, input_desc]() { return executor->Start(inputs, input_desc, loop_count); });

    futures.emplace_back(std::move(future));
  }

  // Push initial tasks
  GELOGD("Start to execute with loops, loop count = %d", loop_count);
  config_.iteration_end = iteration_ + loop_count;
  for (int i = 0; i < config_.num_stages; ++i) {
    StageExecutor::StageTask task_info;
    task_info.stage = i;
    task_info.iteration = iteration_;
    stage_executors_[0]->ExecuteAsync(task_info);
  }

  // Wait for end of iterations
  bool has_error = false;
  for (size_t i = 0; i < stage_executors_.size(); ++i) {
    GELOGD("Start to sync result of executor[%zu]", i);
    auto ret = futures[i].get();
    if (ret != SUCCESS) {
      GELOGE(ret, "[Check][Result][Executor: %zu] Failed to schedule tasks.", i);
      REPORT_INNER_ERROR("E19999", "[Executor: %zu] Failed to schedule tasks.", i);
      has_error = true;
      continue;
    }

    ret = stage_executors_[i]->Synchronize();

    if (ret != SUCCESS) {
      auto model_manager = ModelManager::GetInstance();
      GE_CHECK_NOTNULL(model_manager);
      auto exception_infos = model_manager->GetExceptionInfos();
      if (!exception_infos.empty()) {
        HYBRID_CHK_STATUS_RET(context_.DumpExceptionInfo(exception_infos),
                              "[Execute][GraphInternal] Dump exception info failed.");
      }
      GELOGE(ret, "[Invoke][Synchronize] failed for [Executor: %zu].", i);
      REPORT_CALL_ERROR("E19999", "[Executor: %zu] failed to Synchronize result.", i);
      has_error = true;
      continue;
    }
  }

  // record for profiling analyzer
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[Cleanup] End");

  if (context_.profiler != nullptr) {
    context_.profiler->Dump(std::cout);
  }

  iteration_ = config_.iteration_end;

  if (has_error) {
    GELOGE(FAILED, "[Check][Error]Error occurred while execution.");
    REPORT_INNER_ERROR("E19999", "Error occurred while execution.");
    return FAILED;
  }

  auto last_iter_executor_idx = loop_count % stage_executors_.size();
  GE_CHK_STATUS_RET(stage_executors_[last_iter_executor_idx]->GetOutputs(args.outputs, args.output_desc),
                    "[Get][Outputs]Failed from executor[%zu]", last_iter_executor_idx);
  return SUCCESS;
}

HybridModelPipelineExecutor::~HybridModelPipelineExecutor() {
  GELOGD("~HybridModelPipelineExecutor()");
  for (auto &executor : stage_executors_) {
    (void)executor->context_.profiler.release();
  }
}
}  // namespace hybrid
}  // namespace ge
