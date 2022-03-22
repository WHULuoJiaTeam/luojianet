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

#ifndef GE_SESSION_SESSION_MANAGER_H_
#define GE_SESSION_SESSION_MANAGER_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "framework/common/ge_inner_error_codes.h"
#include "external/ge/ge_api_types.h"
#include "session/inner_session.h"
#include "runtime/base.h"

namespace ge {
using SessionPtr = std::shared_ptr<InnerSession>;

class SessionManager {
 public:
  SessionManager() = default;

  ~SessionManager() = default;

  ///
  /// @ingroup ge_session
  /// @brief initialize session manager
  /// @param [in] options session manager config options
  /// @return Status result of function
  ///
  Status Initialize(const std::map<std::string, std::string> &options);

  ///
  /// @ingroup ge_session
  /// @brief finalize session manager
  /// @return Status result of function
  ///
  Status Finalize();

  ///
  /// @ingroup ge_session
  /// @brief create session
  /// @param [in] options session config options
  /// @param [out] session_id session id
  /// @return Status result of function
  ///
  Status CreateSession(const std::map<std::string, std::string> &options, SessionId &session_id);

  ///
  /// @ingroup ge_session
  /// @brief destroy the session with specific session id
  /// @param [in] session_id session id
  /// @return Status result of function
  ///
  Status DestroySession(SessionId session_id);

  ///
  /// @ingroup ge_session
  /// @brief add a graph to the session with specific session id
  /// @param [in] session_id session id
  /// @param [in] graph_id graph id
  /// @param [in] graph the graph to add
  /// @return Status result of function
  ///
  Status AddGraph(SessionId session_id, uint32_t graph_id, const ge::Graph &graph);

  ///
  /// @ingroup ge_session
  /// @brief add a graph to the session with specific session id and graphOptions
  /// @param [in] session_id session id
  /// @param [in] graph_id graph id
  /// @param [in] graph the graph to add
  /// @param [in] options graph level options
  /// @return Status result of function
  ///
  Status AddGraph(SessionId session_id, uint32_t graph_id, const Graph &graph,
                  const std::map<std::string, std::string> &options);

  ///
  /// @ingroup ge_session
  /// @brief add a copy graph to the session with specific session id and graphOptions
  /// @param [in] session_id session id
  /// @param [in] graph_id graph id
  /// @param [in] graph the graph to add
  /// @param [in] options graph level options
  /// @return Status result of function
  ///
  Status AddGraphWithCopy(SessionId session_id, uint32_t graph_id, const Graph &graph,
                  const std::map<std::string, std::string> &options);

  ///
  /// @ingroup ge_session
  /// @brief run a graph of the session with specific session id
  /// @param [in] session_id session id
  /// @param [in] graph_id graph id
  /// @param [in] inputs input data
  /// @param [out] outputs output data
  /// @return Status result of function
  ///
  Status RunGraph(SessionId session_id, uint32_t graph_id, const std::vector<Tensor> &inputs,
                  std::vector<Tensor> &outputs);

  ///
  /// @ingroup ge_session
  /// @brief run a graph of the session with specific stream asynchronously
  /// @param [in] session_id session id
  /// @param [in] graph_id graph id
  /// @param [in] stream specific stream
  /// @param [in] inputs input data
  /// @param [out] outputs output data
  /// @return Status result of function
  ///
  Status RunGraphWithStreamAsync(SessionId session_id, uint32_t graph_id, rtStream_t stream,
                                 const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs);

  ///
  /// @ingroup ge_session
  /// @brief remove a graph from the session with specific session id
  /// @param [in] session_id session id
  /// @param [in] graph_id graph id
  /// @return Status result of function
  ///
  Status RemoveGraph(SessionId session_id, uint32_t graph_id);

  ///
  /// @ingroup ge_session
  /// @brief get variable value from the session with specific session id
  /// @param [in] session_id session id
  /// @param [in] name op name
  /// @param [out] val out value tensor
  /// @return Status result of function
  ///
  Status GetVariable(SessionId session_id, const std::string &name, Tensor &val);

  ///
  /// @ingroup ge_session
  /// @brief build a graph of the session with specific session id
  /// @param [in] session_id session id
  /// @param [in] graph_id graph id
  /// @param [in] inputs input data
  /// @return Status result of function
  ///
  Status BuildGraph(SessionId session_id, uint32_t graph_id, const std::vector<InputTensorInfo> &inputs);

  Status BuildGraph(SessionId session_id, uint32_t graph_id, const std::vector<ge::Tensor> &inputs);

  ///
  /// @ingroup ge_session
  /// @brief run a graph of the session with specific session id for train asynchronously
  /// @param [in] session_id session id
  /// @param [in] graph_id graph id
  /// @param [in] inputs input data
  /// @return Status result of function
  ///
  Status RunGraphAsync(SessionId session_id, uint32_t graph_id, const std::vector<ge::Tensor> &inputs,
                       RunAsyncCallback callback);

  ///
  /// @ingroup ge_graph
  /// @brief get variables in the session with specific session id
  /// @param [in] session_id: sssion id
  /// @param [in] var_names: variable names
  /// @param [out] var_values: variable values
  /// @return Status result of function
  ///
  Status GetVariables(SessionId session_id, const std::vector<std::string> &var_names,
                      std::vector<Tensor> &var_values);

  ///
  /// @ingroup ge_graph
  /// @brief me register the callback function to get the result of summary or checkpoin
  /// @param [in] session_id session id
  /// @param [in] key: summary or checkpoint
  /// @param [in] callbak: The real callback object of me
  /// @return Status result of function
  ///
  Status RegisterCallBackFunc(
      SessionId session_id, const std::string &key,
      const std::function<Status(uint32_t, const std::map<std::string, ge::Tensor> &)> &callback);
  Status RegisterCallBackFunc(
    SessionId session_id, const std::string &key,
    const std::function<Status(uint32_t, const std::map<AscendString, ge::Tensor> &)> &callback);

  bool IsGraphNeedRebuild(SessionId session_id, uint32_t graph_id);

 private:
  bool HasSession(SessionId session_id);

  Status GetNextSessionId(SessionId &next_session_id);

  Status SetRtContext(SessionId session_id, rtContext_t rtContext);

  std::map<SessionId, SessionPtr> session_manager_map_;
  std::mutex mutex_;
  bool init_flag_ = false;
};
}  // namespace ge

#endif  // GE_SESSION_SESSION_MANAGER_H_
