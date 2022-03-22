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

#include "session/inner_session.h"

#include <map>
#include <memory>
#include <vector>

#include "analyzer/analyzer.h"
#include "adx_datadump_server.h"
#include "common/dump/dump_properties.h"
#include "common/dump/dump_manager.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ge_context.h"
#include "graph/ge_global_options.h"
#include "graph/ge_local_context.h"
#include "common/local_context.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/graph_mem_manager.h"
#include "graph/utils/tensor_adapter.h"
#include "runtime/mem.h"
#include "ir_build/option_utils.h"
#include "common/profiling/profiling_manager.h"

namespace ge {
namespace {
const int32_t kDumpStatus = 0;

Status CheckReuseMemoryOption(const std::map<string, string> &options) {
  auto iter = options.find(OPTION_EXEC_DISABLE_REUSED_MEMORY);
  if (iter != options.end()) {
    if (iter->second == "0") {
      GELOGD("%s=0, reuse memory is open", OPTION_EXEC_DISABLE_REUSED_MEMORY);
    } else if (iter->second == "1") {
      GELOGD("%s=1, reuse memory is close", OPTION_EXEC_DISABLE_REUSED_MEMORY);
    } else {
      GELOGE(PARAM_INVALID, "[CheckReuse][MemoryOption]option %s=%s is invalid", 
          OPTION_EXEC_DISABLE_REUSED_MEMORY, iter->second.c_str());
      REPORT_INNER_ERROR("E19999", "CheckReuseMemoryOption failed because option %s=%s is invalid.", 
          OPTION_EXEC_DISABLE_REUSED_MEMORY, iter->second.c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}
}

static std::mutex mutex_;  // BuildGraph and RunGraph use
bool InnerSession::is_dump_server_inited_ = false;
InnerSession::InnerSession(uint64_t session_id, const std::map<string, string> &options)
    : init_flag_(false), session_id_(session_id), options_(options) {}

Status InnerSession::Initialize() {
  if (init_flag_) {
    GELOGW("[InnerSession:%lu] session already initialize.", session_id_);
    return SUCCESS;
  }

  // If the global options and the session options are duplicated, the session options is preferred.
  auto all_options = options_;
  all_options.insert(GetMutableGlobalOptions().begin(), GetMutableGlobalOptions().end());

  Status ret = CheckReuseMemoryOption(all_options);
  if (ret != SUCCESS) {
    GELOGE(ret, "[CheckReuse][MemoryOption] failed, [InnerSession:%lu].", session_id_);
    REPORT_CALL_ERROR("E19999", "CheckReuseMemoryOption failed, InnerSession=%lu.", session_id_);
    return ret;
  }

  //Check option OP_PRECISION_MODE
  auto iter = all_options.find(ge::OP_PRECISION_MODE);
  if (iter != all_options.end() && !iter->second.empty() && !ge::CheckInputPathValid(iter->second)) {
    REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
        std::vector<std::string>({ge::OP_PRECISION_MODE, iter->second, "path is not found"}));
    GELOGE(PARAM_INVALID, "[Check][OP_PRECISION_MODE] %s not found", iter->second.c_str());
    return FAILED;
  }
  if (iter != all_options.end()) {
    GELOGI("Option set successfully, option_key=%s, option_value=%s",
           ge::OP_PRECISION_MODE.c_str(), iter->second.c_str());
  }
  // Check option modify_mixlist
  if (ge::CheckModifyMixlistParamValid(all_options) != ge::SUCCESS) {
    return FAILED;
  }

  UpdateThreadContext(std::map<std::string, std::string>{});

  // session device id set here
  std::string str_session_device_id;
  if (GetContext().GetOption("ge.session_device_id", str_session_device_id) == SUCCESS) {
    GELOGI("Option session device id has set, value is %s.", str_session_device_id.c_str());

    uint32_t session_device_id = 0;
    try {
      session_device_id = static_cast<uint32_t>(std::stoi(str_session_device_id.c_str()));
      // session device id has priority
      GetContext().SetCtxDeviceId(session_device_id);
    } catch (std::invalid_argument &) {
      GELOGW("session device id %s transform to int failed.", str_session_device_id.c_str());
    } catch (std::out_of_range &) {
      GELOGW("session device id %s transform to int failed.", str_session_device_id.c_str());
    }
  }

  GE_CHK_RT_RET(rtSetDevice(GetContext().DeviceId()));

  DumpProperties dump_properties;
  GE_CHK_STATUS_RET(dump_properties.InitByOptions(), "Init dump properties failed.");
  GE_CHK_STATUS_RET(AddDumpProperties(dump_properties), "[Add][DumpProperties] failed.");

  ret = InnerInitialize();
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][GraphManager] failed, InnerSession:%lu.", session_id_);
    REPORT_CALL_ERROR("E19999", "GraphManager initialize failed, InnerSession:%lu.", session_id_);
    GE_CHK_STATUS(RemoveDumpProperties(), "[Remove][DumpProperties] failed.");
    return ret;
  }

  ret = VarManager::Instance(session_id_)->SetMemoryMallocSize(all_options);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][MemoryMallocSize] failed.");
    REPORT_CALL_ERROR("E19999", "VarManager SetMemoryMallocSize failed, InnerSession:%lu.", session_id_);
    (void)InnerFinalize();
    GE_CHK_STATUS(RemoveDumpProperties(), "[Remove][DumpProperties] failed.");
    GE_CHK_RT(rtDeviceReset(static_cast<int32_t>(GetContext().DeviceId())));
    return ret;
  }

  int32_t version = static_cast<int32_t>(SessionVersion::ClOUD_VERSION);
  const int DEFAULT_DEVICE_ID = 0;
  const int DEFAULT_JOB_ID = 0;
  ret = VarManager::Instance(session_id_)->Init(version, session_id_, DEFAULT_DEVICE_ID, DEFAULT_JOB_ID);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][VarManager] failed.");
    REPORT_CALL_ERROR("E19999", "VarManager init failed, InnerSession:%lu.", session_id_);
    GE_CHK_STATUS(RemoveDumpProperties(), "[Remove][DumpProperties] failed.");
  }
  init_flag_ = true;
  return SUCCESS;
}

Status InnerSession::Finalize() {
  std::lock_guard<std::mutex> lock(resource_mutex_);
  if (!init_flag_) {
    GELOGW("[InnerSession:%lu] session does not initialize.", session_id_);
    return SUCCESS;
  }
  UpdateThreadContext(std::map<std::string, std::string>{});
  Status ret = InnerFinalize();
  if (ret != SUCCESS) {
    // Subsequent code execution is required, so no return is required
    GELOGE(ret, "[Finalize][GraphManager] failed, InnerSession:%lu.", session_id_);
    REPORT_CALL_ERROR("E19999", "GraphManager Finalize failed, InnerSession:%lu.", session_id_);
  }

  init_flag_ = false;
  // release var memory
  GELOGI("VarManager free var memory.");
  (void)VarManager::Instance(session_id_)->FreeVarMemory();

  for (auto memory_type : MemManager::Instance().GetAllMemoryType()) {
    (void)MemManager::Instance().SessionScopeMemInstance(memory_type).Free(session_id_);
  }

  // release analyzer saved info(Session Level)
  Analyzer::GetInstance()->DestroySessionJsonObject(session_id_);

  GE_CHK_RT(rtDeviceReset(static_cast<int32_t>(GetContext().DeviceId())));
  GE_CHK_STATUS_RET(RemoveDumpProperties(), "[Remove][DumpProperties] failed.");

  return ret;
}

Status InnerSession::InnerInitialize() {
  Status ret = model_executor_.Initialize(options_, session_id_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][GraphExecutor] failed, InnerSession:%lu.", session_id_);
    REPORT_CALL_ERROR("E19999", "GraphExecutor initialize failed, InnerSession:%lu.", session_id_);
    GE_CHK_STATUS(RemoveDumpProperties(), "[Remove][DumpProperties] failed.");
    return ret;
  }

  ret = graph_manager_.Initialize(options_, &model_executor_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][GraphManager] failed, InnerSession:%lu.", session_id_);
    REPORT_CALL_ERROR("E19999", "GraphManager initialize failed, InnerSession:%lu.", session_id_);
    GE_CHK_STATUS(RemoveDumpProperties(), "[Remove][DumpProperties] failed.");
    return ret;
  }

  return SUCCESS;
}

Status InnerSession::InnerFinalize() {
  Status ret = graph_manager_.Finalize();
  if (ret != SUCCESS) {
    // Subsequent code execution is required, so no return is required
    GELOGE(ret, "[Finalize][GraphManager] failed, InnerSession:%lu.", session_id_);
    REPORT_CALL_ERROR("E19999", "GraphManager Finalize failed, InnerSession:%lu.", session_id_);
  }

  ret = model_executor_.Finalize();
  if (ret != SUCCESS) {
    // Subsequent code execution is required, so no return is required
    GELOGE(ret, "[Finalize][GraphExecutor] failed, InnerSession:%lu.", session_id_);
    REPORT_CALL_ERROR("E19999", "GraphExecutor Finalize failed, InnerSession:%lu.", session_id_);
  }

  return SUCCESS;
}

Status InnerSession::GetVariable(const std::string &name, Tensor &val) {
  UpdateThreadContext(std::map<std::string, std::string>{});
  return graph_manager_.GetVariable(name, val);
}

Status InnerSession::AddGraph(uint32_t graph_id, const Graph &graph) {
  std::map<std::string, std::string> options;
  auto device_id = GetContext().DeviceId();
  GELOGD("Device id is %u", device_id);
  ProfilingManager::Instance().SetGraphIdToDeviceMap(graph_id, device_id);
  return AddGraph(graph_id, graph, options);
}

Status InnerSession::AddGraph(uint32_t graph_id, const Graph &graph,
                              const std::map<std::string, std::string> &options) {
  std::lock_guard<std::mutex> lock(resource_mutex_);
  if (!init_flag_) {
    GELOGE(GE_SESS_INIT_FAILED, "[Add][Graph] failed because GraphManager not init, InnerSession:%lu, graph_id:%u.",
           session_id_, graph_id);
    REPORT_INNER_ERROR("E19999", "AddGraph failed because GraphManager not init, InnerSession:%lu, graph_id:%u.",
                       session_id_, graph_id);
    return GE_SESS_INIT_FAILED;
  }
  UpdateThreadContext(options);
  Status ret = graph_manager_.AddGraph(graph_id, graph, options, domi::GetContext());
  if (ret != SUCCESS) {
    GELOGE(ret, "[Add][Graph] failed, InnerSession:%lu graphid: %u.", session_id_, graph_id);
    REPORT_CALL_ERROR("E19999", "GraphManager AddGraph failed, InnerSession:%lu graphid: %u.", session_id_, graph_id);
    return ret;
  }

  GELOGI("[InnerSession:%lu] add graph success, graph_id=%u.", session_id_, graph_id);
  return SUCCESS;
}

Status InnerSession::AddGraphWithCopy(uint32_t graph_id, const Graph &graph,
                                      const std::map<std::string, std::string> &options) {
  std::lock_guard<std::mutex> lock(resource_mutex_);
  if (!init_flag_) {
    GELOGE(GE_SESS_INIT_FAILED, "[Add][Graph] failed because GraphManager not init, InnerSession:%lu, graph_id:%u.",
        session_id_, graph_id);
    REPORT_INNER_ERROR("E19999", 
        "AddGraphWithCopy failed because GraphManager not init, InnerSession:%lu, graph_id:%u.", 
        session_id_, graph_id);
    return GE_SESS_INIT_FAILED;
  }
  UpdateThreadContext(options);
  Status ret = graph_manager_.AddGraphWithCopy(graph_id, graph, options, domi::GetContext());
  if (ret != SUCCESS) {
    GELOGE(ret, "[Add][Graph] failed, InnerSession:%lu graphid: %u.", session_id_, graph_id);
    REPORT_CALL_ERROR("E19999", 
        "GraphManager AddGraphWithCopy failed, InnerSession:%lu graphid: %u.", session_id_, graph_id);
    return ret;
  }

  GELOGI("[InnerSession:%lu] add graph success, graph_id=%u.", session_id_, graph_id);
  return SUCCESS;
}

Status InnerSession::RunGraph(uint32_t graph_id, const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) {
  GELOGI("[InnerSession:%lu] run graph on session, graph_id=%u.", session_id_, graph_id);
  if (mutex_.try_lock()) {
    std::lock_guard<std::mutex> lock(mutex_, std::adopt_lock);
    if (!init_flag_) {
      GELOGE(GE_SESS_INIT_FAILED, "[Run][Graph]failed because GraphManager not Init, InnerSession:%lu, graph_id:%u.",
             session_id_, graph_id);
      REPORT_INNER_ERROR("E19999", "RunGraph failed because GraphManager not Init, InnerSession:%lu, graph_id:%u.",
                         session_id_, graph_id);
      return GE_SESS_INIT_FAILED;
    }
    UpdateThreadContext(graph_id);
    vector<GeTensor> geInputs;
    for (auto &item : inputs) {
      geInputs.push_back(TensorAdapter::AsGeTensor(item));
    }
    vector<GeTensor> geOutputs;
    Status ret = graph_manager_.RunGraph(graph_id, geInputs, geOutputs, session_id_);
    domi::GetContext().out_nodes_map.clear();
    domi::GetContext().user_out_nodes.clear();
    if (ret != SUCCESS) {
      GELOGE(ret, "[Run][Graph]failed, InnerSession:%lu graph_id=%u.", session_id_, graph_id);
      REPORT_CALL_ERROR("E19999", 
          "GraphManager RunGraph failed, InnerSession:%lu graph_id=%u.", session_id_, graph_id);
      return ret;
    }
    outputs.clear();
    for (auto &item : geOutputs) {
      outputs.push_back(TensorAdapter::AsTensor(item));
    }

    GELOGI("[InnerSession:%lu] run graph success, graph_id=%u.", session_id_, graph_id);
    return SUCCESS;
  } else {
    GELOGE(GE_SESS_ALREADY_RUNNING, "[Run][Graph]failed, InnerSession:%lu, graph_id=%u.", session_id_, graph_id);
    REPORT_INNER_ERROR("E19999",
                       "RunGraph failed because mutex try_lock false, InnerSession:%lu, graph_id=%u.",
                       session_id_, graph_id);
    return GE_SESS_ALREADY_RUNNING;
  }
}

Status InnerSession::RunGraphWithStreamAsync(uint32_t graph_id, rtStream_t stream,
                                             const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) {
  GELOGI("Run graph with stream, session id = %lu, graph id = %u, stream = %p.",
         session_id_, graph_id, stream);
  if (mutex_.try_lock()) {
    std::lock_guard<std::mutex> lock(mutex_, std::adopt_lock);
    if (!init_flag_) {
      GELOGE(GE_SESS_INIT_FAILED, "[Run][GraphWithStream]failed because GraphManager not Init,"
             "session id = %lu, graph id = %u, stream = %p.", session_id_, graph_id, stream);
      REPORT_INNER_ERROR("E19999", "RunGraphWithStreamAsync failed because GraphManager not Init,"
                         "session id = %lu, graph id = %u, stream = %p.", session_id_, graph_id, stream);
      return GE_SESS_INIT_FAILED;
    }
    UpdateThreadContext(graph_id);
    vector<GeTensor> ge_inputs;
    for (auto &item : inputs) {
      ge_inputs.emplace_back(TensorAdapter::AsGeTensor(item));
    }
    vector<GeTensor> ge_outputs;
    for (auto &item : outputs) {
      ge_outputs.emplace_back(TensorAdapter::AsGeTensor(item));
    }
    Status ret = graph_manager_.RunGraphWithStreamAsync(graph_id, stream, session_id_, ge_inputs, ge_outputs);
    domi::GetContext().out_nodes_map.clear();
    domi::GetContext().user_out_nodes.clear();
    if (ret != SUCCESS) {
      GELOGE(ret, "[Run][GraphWithStreamAsync]failed,"
             "session id = %lu, graph id = %u, stream = %p.", session_id_, graph_id, stream);
      REPORT_CALL_ERROR("E19999", "GraphManager RunGrapWithStreamhAsync failed,"
                        "session id = %lu, graph id = %u, stream = %p.", session_id_, graph_id, stream);
      return ret;
    }

    GELOGI("Run graph with stream success, session id = %lu, graph id = %u, stream = %p.",
           session_id_, graph_id, stream);
    return SUCCESS;
  } else {
    GELOGE(GE_SESS_ALREADY_RUNNING, "[Run][GraphWithStreamAsync]failed because mutex try_lock false,"
           "session id = %lu, graph id = %u, stream = %p.", session_id_, graph_id, stream);
    REPORT_INNER_ERROR("E19999", "[Run][GraphWithStreamAsync]failed failed because mutex try_lock false,"
                       "session id = %lu, graph id = %u, stream = %p.", session_id_, graph_id, stream);
    return GE_SESS_ALREADY_RUNNING;
  }
}

Status InnerSession::RemoveGraph(uint32_t graph_id) {
  std::lock_guard<std::mutex> lock(resource_mutex_);
  if (!init_flag_) {
    GELOGE(GE_SESS_INIT_FAILED,
           "[Remove][Graph] failed because GraphManager not init, InnerSession:%lu, graph_id=%u.",
           session_id_, graph_id);
    REPORT_INNER_ERROR("E19999",
                       "RemoveGraph failed, because GraphManager not init, InnerSession:%lu, graph_id=%u.",
                       session_id_, graph_id);
    return GE_SESS_INIT_FAILED;
  }
  UpdateThreadContext(graph_id);
  Status ret = graph_manager_.RemoveGraph(graph_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Remove][Graph] failed, InnerSession:%lu, graph_id=%u.", session_id_, graph_id);
    REPORT_CALL_ERROR("E19999",
                      "GraphManager RemoveGraph failed, InnerSession:%lu, graph_id=%u.", session_id_, graph_id);
    return ret;
  }

  GELOGI("[InnerSession:%lu] remove graph success, graph_id=%u.", session_id_, graph_id);
  return SUCCESS;
}

Status InnerSession::RegisterCallBackFunc(
    const std::string &key,
    const std::function<Status(uint32_t, const std::map<std::string, ge::Tensor> &)> &callback) {
  std::lock_guard<std::mutex> lock(resource_mutex_);
  if (!init_flag_) {
    GELOGE(GE_SESS_INIT_FAILED,
           "[Register][CallBackFunc] failed because GraphManager not initialize, InnerSession:%lu.", session_id_);
    REPORT_INNER_ERROR("E19999",
                       "RegisterCallBackFunc failed because GraphManager not init, InnerSession:%lu.", session_id_);
    return GE_SESS_INIT_FAILED;
  }
  UpdateThreadContext(std::map<std::string, std::string>{});
  Status ret = graph_manager_.RegisterCallBackFunc(key, callback);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Register][CallBackFunc] failed, InnerSession:%lu register %s.", session_id_, key.c_str());
    REPORT_CALL_ERROR("E19999",
                      "GraphManager RegisterCallBackFunc failed, InnerSession:%lu register %s.",
                      session_id_, key.c_str());
    return ret;
  }

  GELOGI("[InnerSession:%lu] register %s callback function success.", session_id_, key.c_str());
  return SUCCESS;
}

Status InnerSession::RegisterCallBackFunc(
  const std::string &key,
  const std::function<Status(uint32_t, const std::map<AscendString, ge::Tensor> &)> &callback) {
  std::lock_guard<std::mutex> lock(resource_mutex_);
  if (!init_flag_) {
    GELOGE(GE_SESS_INIT_FAILED,
           "[Register][CallBackFunc]failed because GraphManager not initialize, InnerSession:%lu.", session_id_);
    REPORT_INNER_ERROR("E19999",
                       "RegisterCallBackFunc failed because GraphManager not initialize, InnerSession:%lu.",
                       session_id_);
    return GE_SESS_INIT_FAILED;
  }
  UpdateThreadContext(std::map<std::string, std::string>{});
  Status ret = graph_manager_.RegisterCallBackFunc(key, callback);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Register][CallBackFunc] failed, InnerSession:%lu register %s.", session_id_, key.c_str());
    REPORT_CALL_ERROR("E19999",
                      "GraphManager RegisterCallBackFunc failed, InnerSession:%lu register %s.",
                      session_id_, key.c_str());
    return ret;
  }

  GELOGI("[InnerSession:%lu] register %s callback function success.", session_id_, key.c_str());
  return SUCCESS;
}

Status InnerSession::BuildGraph(uint32_t graph_id, const std::vector<InputTensorInfo> &inputs) {
  UpdateThreadContext(graph_id);
  GELOGI("[InnerSession:%lu] build graph on session, graph_id=%u.", session_id_, graph_id);
  std::vector<ge::GeTensor> ge_inputs;
  for (auto const &input : inputs) {
    std::vector<int64_t> input_dims;
    std::transform(input.dims.begin(), input.dims.end(), std::back_inserter(input_dims),
                   [](int64_t x) -> int64_t { return x; });
    GeShape input_shape(input_dims);
    GeTensorDesc input_tensor_desc;
    input_tensor_desc.SetShape(input_shape);
    input_tensor_desc.SetDataType(static_cast<ge::DataType>(input.data_type));
    ge_inputs.emplace_back(input_tensor_desc);
  }
  GeRootModelPtr ge_root_model = nullptr;
  Status ret = graph_manager_.BuildGraph(graph_id, ge_inputs, ge_root_model, session_id_, true);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Build][Graph] failed, InnerSession:%lu graph_id=%u.", session_id_, graph_id);
    REPORT_CALL_ERROR("E19999",
                      "GraphManager BuildGraph failed, InnerSession:%lu graph_id=%u.", session_id_, graph_id);
    return ret;
  }
  GELOGI("[InnerSession:%lu] build graph success, graph_id=%u.", session_id_, graph_id);
  return ret;
}

Status InnerSession::BuildGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs) {
  UpdateThreadContext(graph_id);
  GELOGI("[InnerSession:%lu] build graph on session, graph_id=%u.", session_id_, graph_id);
  std::vector<ge::GeTensor> ge_inputs;
  for (const auto &input : inputs) {
    ge_inputs.emplace_back(TensorAdapter::AsGeTensor(input));
  }
  GeRootModelPtr ge_root_model = nullptr;
  Status ret = graph_manager_.BuildGraph(graph_id, ge_inputs, ge_root_model, session_id_, true);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Build][Graph] failed, InnerSession:%lu graph_id=%u.", session_id_, graph_id);
    REPORT_CALL_ERROR("E19999",
                      "GraphManager BuildGraph failed, InnerSession:%lu graph_id=%u.", session_id_, graph_id);
    return ret;
  }
  GELOGI("[InnerSession:%lu] build graph success, graph_id=%u.", session_id_, graph_id);
  return ret;
}

Status InnerSession::RunGraphAsync(uint32_t graph_id, const std::vector<ge::Tensor> &inputs,
                                   RunAsyncCallback callback) {
  UpdateThreadContext(graph_id);
  GELOGI("[InnerSession:%lu] run graph on session, graph_id=%u.", session_id_, graph_id);
  Status ret = graph_manager_.RunGraphAsync(graph_id, inputs, session_id_, callback);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Run][GraphAsync]failed, InnerSession:%lu graph_id=%u.", session_id_, graph_id);
    REPORT_CALL_ERROR("E19999",
                      "GraphManager RunGraphAsync failed, InnerSession:%lu graph_id=%u.", session_id_, graph_id);
    return ret;
  }
  GELOGI("[InnerSession:%lu] run graph success, graph_id=%u.", session_id_, graph_id);
  return ret;
}
const GraphManager &InnerSession::getGraphManagerObj() const { return graph_manager_; }

void InnerSession::UpdateThreadContext(const std::map<std::string, std::string> &options) {
  GetThreadLocalContext().SetGlobalOption(GetMutableGlobalOptions());
  GetThreadLocalContext().SetSessionOption(options_);
  GetThreadLocalContext().SetGraphOption(options);
  GetContext().SetSessionId(session_id_);
  SetRtSocVersion();
}

void InnerSession::UpdateThreadContext(uint32_t graph_id) {
  auto options = graph_manager_.GetGraphOptions(graph_id);
  if (options == nullptr) {
    GELOGW("graph level options is null.");
    UpdateThreadContext(std::map<std::string, std::string>{});
  } else {
    UpdateThreadContext(*options);
  }
}

bool InnerSession::IsGraphNeedRebuild(uint32_t graph_id) {
  UpdateThreadContext(graph_id);
  return graph_manager_.IsGraphNeedRebuild(graph_id);
}

Status InnerSession::GetAllVariables(std::map<std::string, GeTensorDesc> &all_variables) {
  return VarManager::Instance(session_id_)->GetAllVariables(all_variables);
}

Status InnerSession::GenCheckPointGraph(const std::map<std::string, GeTensorDesc> &all_variables, Graph &graph) {
  return graph_manager_.GenCheckPointGraph(all_variables, graph);
}

Status InnerSession::SaveVariables(const Graph &graph, const std::vector<std::string> &var_names,
                                   const std::vector<Tensor> &outputs, std::vector<Tensor> &var_values) {
  return graph_manager_.SaveVariables(graph, var_names, outputs, var_values);
}

Status InnerSession::AddDumpProperties(const DumpProperties &dump_properties) {
  if (!is_dump_server_inited_) {
    if (dump_properties.IsDumpOpen() || dump_properties.IsOpDebugOpen()) {
      GE_IF_BOOL_EXEC(AdxDataDumpServerInit() != kDumpStatus, 
                      GELOGE(PARAM_INVALID, "[Init][AdxDataDumpServer] failed, session_id:%lu.", session_id_);
                      return PARAM_INVALID)
      GELOGI("Init adx data dump server success");
      is_dump_server_inited_ = true;
    }
  }
  DumpManager::GetInstance().AddDumpProperties(session_id_, dump_properties);
  return SUCCESS;
}

Status InnerSession::RemoveDumpProperties() {
  DumpManager::GetInstance().RemoveDumpProperties(session_id_);
  if (is_dump_server_inited_ && DumpManager::GetInstance().GetDumpPropertiesMap().empty()) {
    GE_IF_BOOL_EXEC(AdxDataDumpServerUnInit() != kDumpStatus,
                    GELOGE(PARAM_INVALID, "[UnInit][AdxDataDumpServer] failed, session_id:%lu.", session_id_);
                    REPORT_INNER_ERROR("E19999", "RemoveDumpProperties failed because AdxDataDumpServerUnInit failed,"
                                       "session_id:%lu", session_id_);
                    return PARAM_INVALID)
    GELOGI("UnInit adx data dump server success");
    is_dump_server_inited_ = false;
  }
  return SUCCESS;
}

void InnerSession::SetRtSocVersion() {
  const auto &global_options = GetMutableGlobalOptions();
  auto it = global_options.find(ge::SOC_VERSION);
  if (it != global_options.end()) {
    const char *soc_version = it->second.c_str();
    rtError_t rt_ret = rtSetSocVersion(soc_version);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("Set soc version %s failed. ret:0x%X", soc_version, rt_ret);
    }
    GELOGI("Set soc version %s success.", soc_version);
  }
}
}  // namespace ge
