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

#ifndef GE_SESSION_INNER_SESSION_H_
#define GE_SESSION_INNER_SESSION_H_

#include <map>
#include <string>
#include <vector>
#include "framework/common/ge_types.h"
#include "external/ge/ge_api_types.h"
#include "graph/manager/graph_manager.h"
#include "graph/execute/model_executor.h"

namespace ge {
class InnerSession {
 public:
  InnerSession(uint64_t session_id, const std::map<string, string> &options);

  ~InnerSession() = default;

  Status Initialize();

  Status AddGraph(uint32_t graph_id, const Graph &graph);

  Status AddGraph(uint32_t graph_id, const Graph &graph, const std::map<std::string, std::string> &options);

  Status AddGraphWithCopy(uint32_t graph_id, const Graph &graph, const std::map<std::string, std::string> &options);

  Status RunGraph(uint32_t graph_id, const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs);

  Status RunGraphWithStreamAsync(uint32_t graph_id, rtStream_t stream, const std::vector<Tensor> &inputs,
                                 std::vector<Tensor> &outputs);

  Status RemoveGraph(uint32_t graph_id);

  Status BuildGraph(uint32_t graph_id, const std::vector<InputTensorInfo> &inputs);

  Status BuildGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs);

  Status RunGraphAsync(uint32_t graph_id, const std::vector<ge::Tensor> &inputs, RunAsyncCallback callback);

  Status Finalize();

  Status GetAllVariables(std::map<std::string, GeTensorDesc> &all_variables);

  Status GenCheckPointGraph(const std::map<std::string, GeTensorDesc> &all_variables, Graph &graph);

  Status SaveVariables(const Graph &graph, const std::vector<std::string> &var_names,
                       const std::vector<Tensor> &outputs, std::vector<Tensor> &var_values);

  Status GetVariable(const std::string &name, Tensor &val);

  Status RegisterCallBackFunc(
      const std::string &key,
      const std::function<Status(uint32_t, const std::map<std::string, ge::Tensor> &)> &callback);

  Status RegisterCallBackFunc(
    const std::string &key,
    const std::function<Status(uint32_t, const std::map<AscendString, ge::Tensor> &)> &callback);

  const GraphManager &getGraphManagerObj() const;

  bool IsGraphNeedRebuild(uint32_t graph_id);

  Status AddDumpProperties(const DumpProperties &dump_properties);

  Status RemoveDumpProperties();

  void SetRtSocVersion();

 private:
  Status InnerInitialize();
  Status InnerFinalize();

  bool init_flag_;
  uint64_t session_id_;
  std::map<string, string> options_;
  GraphManager graph_manager_;
  ModelExecutor model_executor_;
  std::mutex resource_mutex_;  // AddGraph, RemoveGraph and Finalize use
  void UpdateThreadContext(const std::map<std::string, std::string> &options);
  void UpdateThreadContext(uint32_t graph_id);
  static bool is_dump_server_inited_;
};
}  // namespace ge

#endif  // GE_SESSION_INNER_SESSION_H_
