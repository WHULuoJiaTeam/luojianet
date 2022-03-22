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

#include "graph/execute/model_executor.h"

#include "graph/ge_context.h"
#include "graph/debug/ge_attr_define.h"
#include "common/ge_call_wrapper.h"
#include "common/local_context.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/load/graph_loader.h"
#include "graph/load/model_manager/model_manager.h"
#include "common/math/math_util.h"
#include "common/formats/utils/formats_trans_utils.h"

namespace {
constexpr int32_t kBase = 10;
constexpr uint8_t kNeverLoaded = 0;
}

namespace ge {
///
/// @ingroup ge
/// @brief graph executor init
/// @param [in] options user config params
/// @return Status result of function
///
Status ModelExecutor::Initialize(const map<string, string> &options, uint64_t session_id) {
  graph_run_listener_ = MakeShared<GraphModelListener>(sync_run_mutex_, condition_);
  if (graph_run_listener_ == nullptr) {
    REPORT_CALL_ERROR("E19999", "New GraphModelListener fail");
    GELOGE(MEMALLOC_FAILED, "[New][GraphModelListener] failed");
    return MEMALLOC_FAILED;
  }

  const auto model_manager = ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status status = model_manager->EnableExceptionDump(options);
  if (status != SUCCESS) {
    return status;
  }

  session_id_ = session_id;
  train_graph_flag_ = ParseTrainGraphFlag();
  thread_run_flag_.store(true);
  run_thread_ = std::thread(&ModelExecutor::RunThread, this);

  init_flag_ = true;
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief graph executor finalize
/// @return Status result of function
///
Status ModelExecutor::Finalize() {
  if (!init_flag_) {
    GELOGW("ModelExecutor has not been initialized.");
    return SUCCESS;
  }

  StopQueue();
  if (run_thread_.joinable()) {
    run_thread_.join();
  }

  if (graph_executor_.FreeExecuteMemory() != SUCCESS) {
    GELOGW("Graph executor FreeExecuteMemory failed, resources may not be released correctly.");
  }

  ModelManager::GetInstance()->DestroyAicpuSession(session_id_);
  return SUCCESS;
}

// OPTION_GRAPH_RUN_MODE is supposed to be a session-level option, but it used to be set to global-level in the past.
// If can not parse from session, it can parse from global by GetContext().
bool ModelExecutor::ParseTrainGraphFlag() {
  string run_mode;
  if (GetContext().GetOption(OPTION_GRAPH_RUN_MODE, run_mode) == SUCCESS && !run_mode.empty()) {
    if (GraphRunMode(std::strtol(run_mode.c_str(), nullptr, kBase)) >= TRAIN) {
      GELOGI("Graph train flag set.");
      return true;
    }
  }
  return false;
}

void ModelExecutor::AddGraphNode(GraphId graph_id, const GraphNodePtr &graph_node) {
  std::lock_guard<std::mutex> lock(mutex_);
  graph_nodes_.emplace(graph_id, graph_node);
}

void ModelExecutor::RemoveGraphNode(GraphId graph_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  graph_nodes_.erase(graph_id);
}

///
/// @ingroup ge
/// @brief Load mode for graph.
/// @param [in] GeRootModel: root model of graph compiled.
/// @param [in] GraphNode: node of graph.
/// @return Status result of function
///
Status ModelExecutor::LoadGraph(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node) {
  GE_CHECK_NOTNULL(graph_node);
  if (ge_root_model == nullptr) {
    return SUCCESS;
  }

  UpdateLocalOmeContext(graph_node);
  return graph_node->IsAsync() ? ModelLoadAsync(ge_root_model, graph_node) : ModelLoadSync(ge_root_model, graph_node);
}

///
/// @ingroup ge
/// @brief Unload mode for graph.
/// @param [in] GeRootModel: root model of graph compiled.
/// @param [in] graph_id: graph identifier.
/// @return Status result of function
///
Status ModelExecutor::UnloadGraph(const GeRootModelPtr &ge_root_model, uint32_t graph_id) {
  GE_CHECK_NOTNULL(ge_root_model);
  rtError_t rt_ret = rtSetDevice(GetContext().DeviceId());
  if (rt_ret != RT_ERROR_NONE) {
    GELOGW("[GraphExecutor] rtSetDevice failed, modelId=%u, graphId=%u.", ge_root_model->GetModelId(), graph_id);
    return FAILED;
  }

  RemoveGraphNode(graph_id);
  Status ret = UnloadModel(ge_root_model, graph_id);
  if (ret != SUCCESS) {
    GELOGW("[GraphExecutor] unload model failed, graph_id=%u.", graph_id);
  }
  rt_ret = rtDeviceReset(GetContext().DeviceId());
  if (rt_ret != RT_ERROR_NONE) {
    GELOGW("[GraphExecutor] rtDeviceReset failed, graphId=%u.", graph_id);
  }

  return ret;
}

Status ModelExecutor::UnloadModel(const GeRootModelPtr &ge_root_model, uint32_t graph_id) {
  GE_CHECK_NOTNULL(ge_root_model);
  for (size_t i = 0; i < ge_root_model->GetAllModelId().size(); ++i) {
    uint32_t model_id = ge_root_model->GetAllModelId()[i];
    GELOGI("Unload model %u.", model_id);
    Status ret = GraphLoader::UnloadModel(model_id);
    if (ret != SUCCESS) {
      GELOGE(ret, "[GraphExecutor] unload model failed, modelId=%u, graphId=%u.", model_id, graph_id);
      return ret;
    }
  }
  return SUCCESS;
}

void ModelExecutor::StopQueue() {
  thread_run_flag_.store(false);
  run_args_q_.Stop();
}

void ModelExecutor::ReturnError(RunAsyncCallback callback, Status ret, const string &log) {
  StopQueue();
  GELOGE(ret, "%s.", log.c_str());
  std::vector<ge::Tensor> outputs;
  if (callback != nullptr) {
    callback(ret, outputs);
  }
}

void ModelExecutor::UpdateLocalOmeContext(const GraphNodePtr &graph_node) {
  std::lock_guard<std::mutex> lock(mutex_);
  SetLocalOmeContext(graph_node->GetOmeContext());
}

///
/// @ingroup ge
/// @brief Push model execution params to queue.
/// @param [in] RunArgs of for model execution.
/// @return Status result of function
///
Status ModelExecutor::PushGraph(const RunArgs &args) {
  return run_args_q_.Push(args) ? SUCCESS : FAILED;
}

void ModelExecutor::RunThread() {
  ErrorManager::GetInstance().SetStage(error_message::kModelExecute, error_message::kModelExecute);
  if (mmSetCurrentThreadName("GE_Run") != EN_OK) {
    GELOGW("Set thread name failed.");
  }

  RunArgs args;
  while (thread_run_flag_) {
    if (!run_args_q_.Pop(args)) {
      continue;
    }

    GELOGI("[RunThread] A new loop start, graph_id:%u.", args.graph_id);
    ErrorManager::GetInstance().SetErrorContext(args.error_context);
    GetContext().SetSessionId(args.session_id);
    GetThreadLocalContext() = args.context;
    UpdateLocalOmeContext(args.graph_node);

    // parse inputs.dims to vector<vector<uint64_t>> dynamic_dims
    Status ret = ParseInputsDims(args.input_tensor);
    if (ret != SUCCESS) {
      ReturnError(args.callback, ret, "ParseInputsDims failed, thread exit.");
      args.graph_node->Unlock();
      return;
    }

    args.graph_node->UpdateLoadFlag();
    if (!args.graph_node->GetLoadFlag()) {
      ErrorManager::GetInstance().SetStage(error_message::kModelLoad, error_message::kModelLoad);
      args.ge_root_model->SetTrainFlag(train_graph_flag_);
      ret = ModelLoadAsync(args.ge_root_model, args.graph_node);
      if (ret != SUCCESS || args.ge_root_model == nullptr) {
        StopQueue();
        ReturnError(args.callback, ret, "LoadGraphAsync failed, thread exit.");
        args.graph_node->Unlock();
        return;
      }
      // control the times of graph loading in multi-thread scenario
      args.graph_node->DecreaseLoadCount();
      args.graph_node->IncreaseLoadRecord();

      args.graph_node->SetLoadFlag(true);
      GELOGI("LoadGraph[%u], model[%u] success and set LoadFlag to true.", args.graph_node->GetGraphId(),
             args.ge_root_model->GetModelId());
    }

    ErrorManager::GetInstance().SetStage(error_message::kModelExecute, error_message::kModelExecute);
    if (train_graph_flag_) {
      graph_executor_.SetTrainFlag(train_graph_flag_);
    }

    ret = graph_executor_.ExecuteGraphAsync(args.graph_id, args.graph_node->GetGeRootModel(),
                                            args.input_tensor, args.callback);
    args.graph_node->SetRunFlag(false);
    if (ret != SUCCESS) {
      ReturnError(args.callback, ret, "ExecuteGraphAsync failed, thread exit.");
      args.graph_node->Unlock();
      return;
    }
    args.graph_node->Unlock();
    GELOGI("[GraphExecutor] Run graph async success, graph_id=%u.", args.graph_id);
  }
}

///
/// @ingroup ge
/// @brief Run graph for synchronize model.
/// @param [in] graph_node: node of graph.
/// @param [in] graph_id: graph identifier.
/// @param [in] inputs: input data for the graph running.
/// @param [out] outputs: output data of the graph running
/// @return Status result of function
///
Status ModelExecutor::RunGraph(const GraphNodePtr &graph_node, GraphId graph_id,
                                const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs) {
  Status ret = graph_executor_.SetCondition(&sync_run_mutex_, &condition_, graph_run_listener_);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_RUNGRAPH_FAILED, "[Set][Condition] failed, graph_id = %u.", graph_id);
    graph_node->SetRunFlag(false);
    return GE_GRAPH_RUNGRAPH_FAILED;
  }

  if (train_graph_flag_) {
    graph_executor_.SetTrainFlag(train_graph_flag_);
  }
  ret = graph_executor_.ExecuteGraph(graph_id, graph_node->GetGeRootModel(), inputs, outputs);

  graph_node->SetRunFlag(false);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Execute][Graph] failed, graph_id = %u.", graph_id);
    return ret;
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Run graph for NN synchronize model.
/// @param [in] graph_node: node of graph.
/// @param [in] graph_id: graph identifier.
/// @param [in] stream: Stream for model running.
/// @param [in] inputs: input data for the graph running.
/// @param [out] outputs: output data of the graph running
/// @return Status result of function
///
Status ModelExecutor::RunGraphWithStream(const GraphNodePtr &graph_node, GraphId graph_id, rtStream_t stream,
                                          const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs) {
  auto ret = graph_executor_.SetCondition(&sync_run_mutex_, &condition_, graph_run_listener_);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_RUNGRAPH_FAILED, "[Set][Condition] failed, graph id = %u, stream = %p.", graph_id, stream);
    graph_node->SetRunFlag(false);
    return GE_GRAPH_RUNGRAPH_FAILED;
  }

  ret = graph_executor_.ExecuteGraphWithStream(graph_id, stream, graph_node->GetGeRootModel(), inputs, outputs);
  graph_node->SetRunFlag(false);
  graph_node->SetIsSpecificStream(false);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Execute][Graph] With Stream failed, graph id = %u, stream = %p.", graph_id, stream);
    return ret;
  }
  GELOGI("[Run][GraphWithStreamAsync] run graph success, graph id = %u, stream = %p.", graph_id, stream);
  return SUCCESS;
}

Status ModelExecutor::ModelLoadSync(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node) {
  ge_root_model->SetIsSpecificStream(graph_node->IsSpecificStream());
  return ModelLoad(ge_root_model, graph_node, graph_run_listener_);
}

Status ModelExecutor::ModelLoadAsync(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node) {
  auto listener = MakeShared<RunAsyncListener>();
  GE_CHECK_NOTNULL(listener);
  return ModelLoad(ge_root_model, graph_node, listener);
}

Status ModelExecutor::ModelLoad(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node,
                                 const std::shared_ptr<ModelListener> &listener) {
  ge_root_model->SetTrainFlag(train_graph_flag_);
  bool is_unknown_shape = false;
  GE_CHK_STATUS_RET(ge_root_model->CheckIsUnknownShape(is_unknown_shape));
  if (!is_unknown_shape) {
    if (getenv(kEnvGeuseStaticMemory) != nullptr) {
      GELOGI("[LoadGraph] GE_USE_STATIC_MEMORY is seted.");
    } else {
      auto root_graph = ge_root_model->GetRootGraph();
      GE_CHECK_NOTNULL(root_graph);
      auto name_to_model = ge_root_model->GetSubgraphInstanceNameToModel();
      GeModelPtr ge_model = name_to_model[root_graph->GetName()];
      GE_CHK_STATUS_RET(CheckAndReleaseMemory(ge_model, graph_node));
    }
  }
  GE_TIMESTAMP_START(LoadModelOnline);
  uint32_t model_id = INVALID_MODEL_ID;
  Status ret = GraphLoader::LoadModelOnline(model_id, ge_root_model, listener);
  GE_TIMESTAMP_EVENT_END(LoadModelOnline, "GraphLoader::LoadModelOnline");
  if (ret != SUCCESS) {
    GELOGE(ret, "[Load][ModelOnline] Failed, model_id:%u", model_id);
    graph_node->SetRunFlag(false);
    return ret;
  }
  graph_node->SetLoadFlag(true);
  ge_root_model->SetModelId(model_id);
  graph_node->SetGeRootModel(ge_root_model);
  AddGraphNode(graph_node->GetGraphId(), graph_node);
  return SUCCESS;
}

void ModelExecutor::ReleaseMemory(const GeModelPtr &ge_model, const GraphNodePtr &graph_node,
                                   const std::vector<uint32_t> &model_ids, uint32_t graph_id, uint64_t session_id) {
  rtError_t rt_ret = rtSetDevice(GetContext().DeviceId());
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtSetDevice failed, device_id:%u", GetContext().DeviceId());
    GELOGE(RT_FAILED, "[Call][RtSetDevice] failed, device_id=%u.", GetContext().DeviceId());
    return;
  }
  for (auto model_id : model_ids) {
    uint64_t max_memory_size = 0;
    Status result = GraphLoader::GetMaxUsedMemory(model_id, max_memory_size);
    if (result != SUCCESS) {
      continue;
    }
    GELOGI("try to UnloadGraph[%u], model[%u] which MaxUsedMemory[%lu].", graph_id, model_id, max_memory_size);
    if (model_ids.size() > 1) {
      result = ge_model->GetSessionId(model_id, session_id);
      if (result != SUCCESS) {
        GELOGW("[GraphExecutor:] get session failed when dynamic memory, modelId=%u, graphId=%u.", model_id,
               graph_id);
        continue;
      }
    }
    result = GraphLoader::DestroyAicpuKernel(session_id, model_id, 0);
    if (result != SUCCESS) {
      GELOGW("[GraphExecutor:] destroy aicpu kernel failed when dynamic memory, modelId=%u, graphId=%u.", model_id,
             graph_id);
    }
    result = GraphLoader::UnloadModel(model_id);
    if (result != SUCCESS) {
      GELOGW("[GraphExecutor:] unload model failed, modelId=%u, graphId=%u.", model_id, graph_id);
    }
    GELOGI("UnloadGraph[%u], model[%u] success.", graph_id, model_id);
  }
  graph_node->SetLoadFlag(false);
  // Allow model to be loaded agagin without adding graph again
  graph_node->SetLoadCount(graph_node->GetLoadRecord());
  graph_node->SetLoadRecord(kNeverLoaded);
  GeRootModelPtr ge_root_model = graph_node->GetGeRootModel();
  if (ge_root_model == nullptr) {
    GELOGW("ge_root_model is null, graph_id:%u", graph_id);
    return;
  }
  ge_root_model->ClearAllModelId();
  rt_ret = rtDeviceReset(GetContext().DeviceId());
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtDeviceReset failed, device_id:%u", GetContext().DeviceId());
    GELOGE(RT_FAILED, "[Call][RtDeviceReset] failed, device_id:%u.", GetContext().DeviceId());
    return;
  }
}

Status ModelExecutor::CheckAndReleaseMemory(const GeModelPtr &ge_model, const GraphNodePtr &graph_node) {
  GELOGI("graph_id[%u]", graph_node->GetGraphId());
  int64_t free_memory = 0;
  Status result = GraphLoader::GetMemoryInfo(free_memory);
  if (result != SUCCESS) {
    return result;
  }

  int64_t value = 0;
  int64_t memory_size = AttrUtils::GetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, value) ? value : 0;
  int64_t weight_size = AttrUtils::GetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, value) ? value : 0;
  int64_t session_id = AttrUtils::GetInt(ge_model, MODEL_ATTR_SESSION_ID, value) ? value : 0;

  GELOGI("Graph[%u] need memory_size[%ld], weight_size[%ld], Device[%u] free_memory_size[%ld]",
         graph_node->GetGraphId(), memory_size, weight_size, GetContext().DeviceId(), free_memory);
  if (CheckInt64AddOverflow(memory_size, weight_size) != SUCCESS) {
    REPORT_INNER_ERROR("E19999", "memory_size:%ld and weight_size:%ld will overflow after add, check invalid",
                       memory_size, weight_size);
    GELOGE(INTERNAL_ERROR, "[Check][Param] memory_size:%ld and weight_size:%ld will overflow after add",
           memory_size, weight_size);
    return INTERNAL_ERROR;
  }
  if (free_memory >= (memory_size + weight_size)) {
    return SUCCESS;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto &it : graph_nodes_) {
    auto graph_id = it.second->GetGraphId();
    auto model = it.second->GetGeRootModel();
    if (model == nullptr) {
      continue;
    }
    auto model_id = model->GetModelId();
    auto model_ids = model->GetAllModelId();
    // unload model not release
    bool is_unknown_shape = false;
    GE_CHK_STATUS_RET(model->CheckIsUnknownShape(is_unknown_shape));
    if (is_unknown_shape) {
      GELOGD("model_id[%u] graph_id[%u] is unknown model, not release memory", model_id, graph_id);
      continue;
    }
    // not loaded,no need unload
    if (!it.second->GetLoadFlag()) {
      GELOGI("CheckAndReleaseMemory graph[%u] has not been loaded.", graph_id);
      continue;
    }
    ReleaseMemory(ge_model, it.second, model_ids, graph_id, static_cast<uint64_t>(session_id));
  }

  return SUCCESS;
}

void ModelExecutor::ParseInputsDimsForData(const std::vector<ge::Tensor> &input_tensor) {
  GELOGD("Start parse input dims from data.");
  for (size_t i = 0; i < input_tensor.size(); ++i) {
    const TensorDesc &tensor_desc = input_tensor[i].GetTensorDesc();
    const Shape &shape = tensor_desc.GetShape();
    const auto &shape_dims = shape.GetDims();
    GELOGD("Input tensor dims is %s.", formats::JoinToString(shape_dims).c_str());
    GetLocalOmeContext().user_real_input_dims.emplace_back(shape_dims);
  }
}

Status ModelExecutor::ParseInputsDimsForGetNextNoSinkAndData(const vector<NodePtr> &dynamic_nodes,
                                                             const std::vector<ge::Tensor> &input_tensor) {
  GELOGD("Start parse inputs dims when coexist data and getnext sink.");
  for (size_t i = 0; i < dynamic_nodes.size(); ++i) {
    auto op_desc = dynamic_nodes.at(i)->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    GeAttrValue::INT index = 0;
    if (!(AttrUtils::GetInt(op_desc, ATTR_NAME_INDEX, index))) {
      REPORT_CALL_ERROR("E19999", "Get Attr:%s from op:%s(%s) fail", ATTR_NAME_INDEX.c_str(),
                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(PARAM_INVALID, "[Get][Attr] %s from op:%s(%s) fail", ATTR_NAME_INDEX.c_str(),
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return PARAM_INVALID;
    }
    if (static_cast<size_t>(index) > input_tensor.size()) {
      REPORT_INNER_ERROR("E19999", "Attr:%s in op:%s(%s) value:%ld > param input_tensor.size:%zu, "
                                   "check invalid", ATTR_NAME_INDEX.c_str(),
                         op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                         index, input_tensor.size());
      GELOGE(PARAM_INVALID, "[Check][Param] Attr:%s in op:%s(%s) value:%ld > param input_tensor.size:%zu",
             ATTR_NAME_INDEX.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str(),
             index, input_tensor.size());
      return PARAM_INVALID;
    }

    const TensorDesc &tensor_desc = input_tensor[i].GetTensorDesc();
    const Shape &shape = tensor_desc.GetShape();
    const auto &shape_dims = shape.GetDims();
    GELOGI("Shape dims of %zu data is %s.", index, formats::JoinToString(shape_dims).c_str());
    GetLocalOmeContext().user_real_input_dims.emplace_back(std::move(shape_dims));
  }
  return SUCCESS;
}

Status ModelExecutor::ParseInputsDims(const std::vector<ge::Tensor> &input_tensor) {
  GELOGI("Start parse input dims of %zu input tensor.", input_tensor.size());
  GetLocalOmeContext().user_real_input_dims.clear();
  if (GetLocalOmeContext().dynamic_node_type.empty()) {
    return SUCCESS;
  }

  const vector<NodePtr> &data_nodes = GetLocalOmeContext().data_nodes;
  const vector<NodePtr> &getnext_nosink_nodes = GetLocalOmeContext().getnext_nosink_nodes;
  GELOGD("Data nodes count is %zu, getnext nosink nodes count is %zu.", data_nodes.size(),
         getnext_nosink_nodes.size());
  if (GetLocalOmeContext().dynamic_node_type == DATA) {
    if (getnext_nosink_nodes.empty()) {
      // just data or data+getnext_sink
      ParseInputsDimsForData(input_tensor);
    } else {
      // data+getnext_nosink, but only need to get shape_dims of data
      if (ParseInputsDimsForGetNextNoSinkAndData(data_nodes, input_tensor) != SUCCESS) {
        GELOGE(PARAM_INVALID, "[Parse][Dims] from data failed, when data coexist with getnext nosink.");
        return PARAM_INVALID;
      }
    }
  } else {
    if (getnext_nosink_nodes.empty()) {
      // just getnext_sink or getnext_sink+data, need to get shape_dims from aicpu op
      GELOGI("Need to get dims from aicpu op: GETDYNAMICDIMS.");
      return SUCCESS;
    } else {
      if (data_nodes.empty()) {
        // just getnext_nosink
        ParseInputsDimsForData(input_tensor);
      } else {
        // getnext_nosink + data, but only need to get shape_dims of getnext_nosink
        if (ParseInputsDimsForGetNextNoSinkAndData(getnext_nosink_nodes, input_tensor) != SUCCESS) {
          GELOGE(PARAM_INVALID, "[Parse][Dims] from getnext nosink failed, when data coexist with getnext nosink");
          return PARAM_INVALID;
        }
      }
    }
  }

  GELOGI("Parse %zu inputs dims success.", GetLocalOmeContext().user_real_input_dims.size());
  return SUCCESS;
}
} // namespace ge
