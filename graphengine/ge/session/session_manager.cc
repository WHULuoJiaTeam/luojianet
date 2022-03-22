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

#include "session/session_manager.h"
#include <memory>
#include <utility>
#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ge_context.h"
#include "graph/manager/util/rt_context_util.h"

using std::map;
using std::string;
using std::vector;

namespace ge {
Status SessionManager::Initialize(const std::map<std::string, std::string> &options) {
  if (init_flag_) {
    GELOGW("Session Manager has been initialized.");
    return SUCCESS;
  }
  init_flag_ = true;
  return SUCCESS;
}

Status SessionManager::Finalize() {
  if (!init_flag_) {
    GELOGW("Session Manager has not been initialized.");
    return SUCCESS;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto iter = session_manager_map_.begin(); iter != session_manager_map_.end(); ++iter) {
    (void)iter->second->Finalize();
  }
  session_manager_map_.clear();
  init_flag_ = false;
  return SUCCESS;
}

Status SessionManager::SetRtContext(SessionId session_id, rtContext_t rt_context) {
  GELOGI("set rt_context RT_CTX_NORMAL_MODE, device id:%u.", GetContext().DeviceId());
  GE_CHK_RT_RET(rtCtxCreate(&rt_context, RT_CTX_NORMAL_MODE, static_cast<int32_t>(GetContext().DeviceId())));
  GE_CHK_RT_RET(rtCtxSetCurrent(rt_context));
  RtContextUtil::GetInstance().AddRtContext(session_id, rt_context);
  return SUCCESS;
}

Status SessionManager::CreateSession(const std::map<std::string, std::string> &options, SessionId &session_id) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT, "[Create][Session]fail for Session manager is not initialized.");
    REPORT_INNER_ERROR("E19999", "CreateSession fail for Session manager is not initialized.");
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  SessionId next_session_id = 0;

  std::lock_guard<std::mutex> lock(mutex_);
  Status nextSessionIdRet = GetNextSessionId(next_session_id);
  if (nextSessionIdRet != SUCCESS) {
    return nextSessionIdRet;
  }

  SessionPtr sessionPtr = MakeShared<InnerSession>(next_session_id, options);
  if (sessionPtr == nullptr) {
    return MEMALLOC_FAILED;
  }
  Status ret = sessionPtr->Initialize();
  if (ret != SUCCESS) {
    return ret;
  }

  (void)session_manager_map_.emplace(std::pair<SessionId, SessionPtr>(next_session_id, sessionPtr));
  session_id = next_session_id;

  // create a context
  ret = SetRtContext(session_id, rtContext_t());

  return ret;
}

Status SessionManager::DestroySession(SessionId session_id) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT,
           "[Destroy][Session]fail for Session manager is not initialized, session_id:%lu.", session_id);
    REPORT_INNER_ERROR("E19999", "DestroySession fail for Session manager is not initialized, session_id:%lu.",
                       session_id);
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  std::map<SessionId, SessionPtr>::iterator it = session_manager_map_.find(session_id);
  if (it == session_manager_map_.end()) {
    return GE_SESSION_NOT_EXIST;
  }

  // Unified destruct rt_context
  RtContextUtil::GetInstance().DestroyRtContexts(session_id);

  SessionPtr innerSession = it->second;
  Status ret = innerSession->Finalize();
  if (ret != SUCCESS) {
    return ret;
  }
  (void)session_manager_map_.erase(session_id);
  return ret;
}

Status SessionManager::GetVariable(SessionId session_id, const std::string &name, Tensor &val) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT,
           "[Get][Variable]fail for Session manager is not initialized, session_id:%lu, input_name:%s.",
           session_id, name.c_str());
    REPORT_INNER_ERROR("E19999",
                       "GetVariable fail for Session manager is not initialized, session_id:%lu, input_name:%s.",
                       session_id, name.c_str());
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  SessionPtr innerSession = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<SessionId, SessionPtr>::iterator it = session_manager_map_.find(session_id);
    if (it == session_manager_map_.end()) {
      return GE_SESSION_NOT_EXIST;
    } else {
      innerSession = it->second;
    }
  }
  return innerSession->GetVariable(name, val);
}

Status SessionManager::AddGraph(SessionId session_id, uint32_t graph_id, const Graph &graph) {
  std::map<std::string, std::string> options;
  return AddGraph(session_id, graph_id, graph, options);
}

Status SessionManager::AddGraph(SessionId session_id, uint32_t graph_id, const Graph &graph,
                                const std::map<std::string, std::string> &options) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT,
           "[Add][Graph]fail for Session manager is not initialized, session_id:%lu, graph_id:%u.",
           session_id, graph_id);
    REPORT_INNER_ERROR("E19999", "AddGraph fail for Session manager is not initialized, session_id:%lu, graph_id:%u.",
                       session_id, graph_id);
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  SessionPtr innerSession = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<SessionId, SessionPtr>::iterator it = session_manager_map_.find(session_id);
    if (it == session_manager_map_.end()) {
      return GE_SESSION_NOT_EXIST;
    } else {
      innerSession = it->second;
    }
    auto compute_graph = GraphUtils::GetComputeGraph(graph);
    GE_CHECK_NOTNULL(compute_graph);
    std::string session_graph_id = std::to_string(session_id) + "_" + std::to_string(graph_id);
    if (!AttrUtils::SetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
      GELOGW("Set graph session_graph_id attr failed.");
    } else {
      GELOGD("Set graph session_graph_id attr to [%s]", session_graph_id.c_str());
    }
    for (auto graph : compute_graph->GetAllSubgraphs()) {
      AttrUtils::SetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
    }
  }
  return innerSession->AddGraph(graph_id, graph, options);
}

Status SessionManager::AddGraphWithCopy(SessionId session_id, uint32_t graph_id, const Graph &graph,
                                        const std::map<std::string, std::string> &options) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT,
           "[Add][GraphWithCopy]fail for Session manager is not initialized, session_id:%lu, graph_id:%u.",
           session_id, graph_id);
    REPORT_INNER_ERROR("E19999",
                       "AddGraphWithCopy fail for Session manager is not initialized, session_id:%lu, graph_id:%u",
                       session_id, graph_id);
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  SessionPtr innerSession = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<SessionId, SessionPtr>::iterator it = session_manager_map_.find(session_id);
    if (it == session_manager_map_.end()) {
      return GE_SESSION_NOT_EXIST;
    } else {
      innerSession = it->second;
    }
    auto compute_graph = GraphUtils::GetComputeGraph(graph);
    GE_CHECK_NOTNULL(compute_graph);
    std::string session_graph_id = std::to_string(session_id) + "_" + std::to_string(graph_id);
    if (!AttrUtils::SetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
      GELOGW("Set graph session_graph_id attr failed.");
    } else {
      GELOGD("Set graph session_graph_id attr to [%s]", session_graph_id.c_str());
    }
    for (auto graph : compute_graph->GetAllSubgraphs()) {
      AttrUtils::SetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
    }
  }
  return innerSession->AddGraphWithCopy(graph_id, graph, options);
}

Status SessionManager::RunGraph(SessionId session_id, uint32_t graph_id, const std::vector<Tensor> &inputs,
                                std::vector<Tensor> &outputs) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT,
           "[Run][Graph]fail for Session manager is not initialized, session_id:%lu, graph_id:%u.",
           session_id, graph_id);
    REPORT_INNER_ERROR("E19999",
                       "RunGraph fail for Session manager is not initialized, session_id:%lu, graph_id:%u.",
                       session_id, graph_id);
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  SessionPtr innerSession = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<SessionId, SessionPtr>::iterator it = session_manager_map_.find(session_id);
    if (it == session_manager_map_.end()) {
      return GE_SESSION_NOT_EXIST;
    } else {
      innerSession = it->second;
    }
  }
  return innerSession->RunGraph(graph_id, inputs, outputs);
}

Status SessionManager::RunGraphWithStreamAsync(SessionId session_id,
                                               uint32_t graph_id,
                                               rtStream_t stream,
                                               const std::vector<Tensor> &inputs,
                                               std::vector<Tensor> &outputs) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT,
           "[RunWithStream][Graph]Session manager is not initialized,"
           "session id = %lu, graph id = %u, stream = %p.", session_id, graph_id, stream);
    REPORT_INNER_ERROR("E19999",
        "RunGraphWithStreamAsync fail for Session manager is not initialized,"
        "session id = %lu, graph id = %u, stream = %p.", session_id, graph_id, stream);
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  SessionPtr innerSession = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<SessionId, SessionPtr>::iterator it = session_manager_map_.find(session_id);
    if (it == session_manager_map_.end()) {
      return GE_SESSION_NOT_EXIST;
    } else {
      innerSession = it->second;
    }
  }
  return innerSession->RunGraphWithStreamAsync(graph_id, stream, inputs, outputs);
}

Status SessionManager::RemoveGraph(SessionId session_id, uint32_t graph_id) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT,
           "[Remove][Graph]fail for Session manager is not initialized, session_id:%lu graph_id:%u.",
           session_id, graph_id);
    REPORT_INNER_ERROR("E19999",
                       "RemoveGraph fail for Session manager is not initialized, session_id:%lu graph_id:%u.",
                       session_id, graph_id);
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  SessionPtr innerSession = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<SessionId, SessionPtr>::iterator it = session_manager_map_.find(session_id);
    if (it == session_manager_map_.end()) {
      return GE_SESSION_NOT_EXIST;
    } else {
      innerSession = it->second;
    }
  }
  return innerSession->RemoveGraph(graph_id);
}

bool SessionManager::HasSession(SessionId session_id) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT,
           "[Has][Session]fail for Session manager is not initialized, session_id:%lu.", session_id);
    REPORT_INNER_ERROR("E19999",
                       "HasSession fail for Session manager is not initialized, session_id:%lu.", session_id);
    return false;
  }
  return session_manager_map_.find(session_id) != session_manager_map_.end();
}

Status SessionManager::GetNextSessionId(SessionId &next_session_id) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT, "[Get][NextSessionId]fail for Session manager is not initialized.");
    REPORT_INNER_ERROR("E19999",  "GetNextSessionId fail for Session manager is not initialized.");
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  static SessionId session_id = 0;

  next_session_id = session_id++;
  return SUCCESS;
}

Status SessionManager::RegisterCallBackFunc(
    SessionId session_id, const std::string &key,
    const std::function<Status(uint32_t, const std::map<std::string, ge::Tensor> &)> &callback) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT,
           "[Register][CallBackFunc]fail for Session manager is not initialized, session_id:%lu, input_key:%s.",
           session_id, key.c_str());
    REPORT_INNER_ERROR("E19999", "RegisterCallBackFunc fail for Session manager is not initialized,"
                       "session_id:%lu, input_key:%s.", session_id, key.c_str());
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  SessionPtr innerSession = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<SessionId, SessionPtr>::iterator it = session_manager_map_.find(session_id);
    if (it == session_manager_map_.end()) {
      return GE_SESSION_NOT_EXIST;
    } else {
      innerSession = it->second;
    }
  }
  return innerSession->RegisterCallBackFunc(key, callback);
}

Status SessionManager::RegisterCallBackFunc(
  SessionId session_id, const std::string &key,
  const std::function<Status(uint32_t, const std::map<AscendString, ge::Tensor> &)> &callback) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT,
           "[Register][CallBackFunc]fail for Session manager is not initialized, session_id:%lu, input_key:%s.",
           session_id, key.c_str());
    REPORT_INNER_ERROR("E19999", "RegisterCallBackFunc fail for Session manager is not initialized,"
                       "session_id:%lu, input_key:%s.", session_id, key.c_str());
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  SessionPtr innerSession = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<SessionId, SessionPtr>::iterator it = session_manager_map_.find(session_id);
    if (it == session_manager_map_.end()) {
      return GE_SESSION_NOT_EXIST;
    } else {
      innerSession = it->second;
    }
  }
  return innerSession->RegisterCallBackFunc(key, callback);
}

Status SessionManager::BuildGraph(SessionId session_id, uint32_t graph_id, const std::vector<InputTensorInfo> &inputs) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT, "[Build][Graph]fail for Session manager is not initialized,"
           "session_id:%lu, graph_id:%u.", session_id, graph_id);
    REPORT_INNER_ERROR("E19999", "BuildGraph fail for Session manager is not initialized,"
                       "session_id:%lu, graph_id:%u.", session_id, graph_id);
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  SessionPtr innerSession = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<SessionId, SessionPtr>::iterator it = session_manager_map_.find(session_id);
    if (it == session_manager_map_.end()) {
      return GE_SESSION_NOT_EXIST;
    } else {
      innerSession = it->second;
    }
  }
  return innerSession->BuildGraph(graph_id, inputs);
}

Status SessionManager::BuildGraph(SessionId session_id, uint32_t graph_id, const std::vector<ge::Tensor> &inputs) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT, "[Build][Graph]fail for Session manager is not initialized,"
           "session_id:%lu, graph_id:%u.", session_id, graph_id);
    REPORT_INNER_ERROR("E19999", "BuildGraph fail for Session manager is not initialized,"
                       "session_id:%lu, graph_id:%u.", session_id, graph_id);
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  SessionPtr innerSession = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<SessionId, SessionPtr>::iterator it = session_manager_map_.find(session_id);
    if (it == session_manager_map_.end()) {
      return GE_SESSION_NOT_EXIST;
    } else {
      innerSession = it->second;
    }
  }
  return innerSession->BuildGraph(graph_id, inputs);
}

Status SessionManager::RunGraphAsync(SessionId session_id, uint32_t graph_id,
                                     const std::vector<ge::Tensor> &inputs, RunAsyncCallback callback) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT,
           "[AsyncRun][Graph]fail for Session manager is not initialized, session_id:%lu, graph_id:%u.",
           session_id, graph_id);
    REPORT_INNER_ERROR("E19999",
                       "RunGraphAsync fail for Session manager is not initialized, session_id:%lu, graph_id:%u.",
                       session_id, graph_id);
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  SessionPtr innerSession = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<SessionId, SessionPtr>::iterator it = session_manager_map_.find(session_id);
    if (it == session_manager_map_.end()) {
      return GE_SESSION_NOT_EXIST;
    } else {
      innerSession = it->second;
    }
  }
  return innerSession->RunGraphAsync(graph_id, inputs, callback);
}

Status SessionManager::GetVariables(SessionId session_id, const std::vector<std::string> &var_names,
                                    std::vector<Tensor> &var_values) {
  // step 0: init session manager
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT,
           "[Get][Variables]fail for Session manager is not initialized, session_id:%lu", session_id);
    REPORT_INNER_ERROR("E19999",
                       "GetVariables fail for Session manager is not initialized, session_id:%lu", session_id);
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  SessionPtr innerSession = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<SessionId, SessionPtr>::iterator it = session_manager_map_.find(session_id);
    if (it == session_manager_map_.end()) {
      return GE_SESSION_NOT_EXIST;
    } else {
      innerSession = it->second;
    }
  }

  // step 1: get all variable
  std::map<std::string, GeTensorDesc> all_variables;
  Status ret = innerSession->GetAllVariables(all_variables);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Get][AllVariables]failed.");
    return FAILED;
  }

  // srep 2: create check point graph
  Graph graph = Graph("checkpoint");
  ret = innerSession->GenCheckPointGraph(all_variables, graph);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[GenCheck][PointGraph] failed.");
    return FAILED;
  }

  // step 3: run check point graph
  uint32_t graph_id = GetCurrentSecondTimestap();
  ret = AddGraph(session_id, graph_id, graph);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Add][Graph] failed.");
    return FAILED;
  }

  vector<Tensor> inputs;
  vector<Tensor> outputs;
  ret = RunGraph(session_id, graph_id, inputs, outputs);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Run][Graph] failed.");
    return FAILED;
  }

  // step 4: save variables
  ret = innerSession->SaveVariables(graph, var_names, outputs, var_values);
  GELOGD("[SessionManager] outputs size is [%zu], var values size is [%zu].", outputs.size(), var_values.size());

  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Save][Variables] failed.");
    return FAILED;
  }

  // step 5: remove graph
  ret = innerSession->RemoveGraph(graph_id);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Remove][Graph] failed.");
    return FAILED;
  }
  return ret;
}

bool SessionManager::IsGraphNeedRebuild(SessionId session_id, uint32_t graph_id) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT,
           "[Check][GraphNeedRebuild]fail for Session manager is not initialized, session_id:%lu, graph_id:%u.",
           session_id, graph_id);
    REPORT_INNER_ERROR("E19999",
                       "IsGraphNeedRebuild fail for Session manager is not initialized, session_id:%lu, graph_id:%u.",
                       session_id, graph_id);
    return true;
  }
  SessionPtr innerSession = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = session_manager_map_.find(session_id);
    if (it == session_manager_map_.end()) {
      GELOGE(GE_SESSION_NOT_EXIST, "[Find][InnerSession] fail for %lu does not exists", session_id);
      REPORT_INNER_ERROR("E19999",
                         "IsGraphNeedRebuild fail for InnerSession is not exists, session_id:%lu, graph_id:%u.",
                         session_id, graph_id);
      return true;
    } else {
      innerSession = it->second;
    }
  }
  return innerSession->IsGraphNeedRebuild(graph_id);
}
}  // namespace ge
