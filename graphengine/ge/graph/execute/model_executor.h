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
#ifndef GE_GRAPH_EXECUTE_MODEL_EXECUTOR_H
#define GE_GRAPH_EXECUTE_MODEL_EXECUTOR_H

#include <thread>

#include "common/executor.h"
#include "graph/execute/graph_execute.h"

namespace ge {
class ModelExecutor : public Executor {
 public:
  ///
  /// @ingroup ge
  /// @brief graph executor init
  /// @param [in] options user config params
  /// @return Status result of function
  ///
  Status Initialize(const map<string, string> &options, uint64_t session_id);

  ///
  /// @ingroup ge
  /// @brief graph executor finalize
  /// @return Status result of function
  ///
  Status Finalize();

  ///
  /// @ingroup ge
  /// @brief Load mode for graph.
  /// @param [in] GeRootModel: root model of graph compiled.
  /// @param [in] GraphNode: node of graph.
  /// @return Status result of function
  ///
  Status LoadGraph(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node);

  ///
  /// @ingroup ge
  /// @brief Unload mode for graph.
  /// @param [in] GeRootModel: root model of graph compiled.
  /// @param [in] graph_id: graph identifier.
  /// @return Status result of function
  ///
  Status UnloadGraph(const GeRootModelPtr &ge_root_model, uint32_t graph_id);

  ///
  /// @ingroup ge
  /// @brief Push model execution params to queue.
  /// @param [in] RunArgs of for model execution.
  /// @return Status result of function
  ///
  Status PushGraph(const RunArgs &args);

  ///
  /// @ingroup ge
  /// @brief Run graph for synchronize model.
  /// @param [in] graph_node: node of graph.
  /// @param [in] graph_id: graph identifier.
  /// @param [in] inputs: input data for the graph running.
  /// @param [out] outputs: output data of the graph running
  /// @return Status result of function
  ///
  Status RunGraph(const GraphNodePtr &graph_node, GraphId graph_id,
                  const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs);

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
  Status RunGraphWithStream(const GraphNodePtr &graph_node, GraphId graph_id, rtStream_t stream,
                            const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs);

 private:
  bool ParseTrainGraphFlag();

  void AddGraphNode(GraphId graph_id, const GraphNodePtr &graph_node);
  void RemoveGraphNode(GraphId graph_id);

  Status ModelLoadSync(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node);
  Status ModelLoadAsync(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node);
  Status ModelLoad(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node,
                   const std::shared_ptr<ModelListener> &listener);

  Status UnloadModel(const GeRootModelPtr &ge_root_model, uint32_t graph_id);

  void ReleaseMemory(const GeModelPtr &ge_model, const GraphNodePtr &graph_node, const std::vector<uint32_t> &model_ids,
                     uint32_t graph_id, uint64_t session_id);
  Status CheckAndReleaseMemory(const GeModelPtr &ge_model, const GraphNodePtr &graph_node);

  void UpdateLocalOmeContext(const GraphNodePtr &graph_node);

  void RunThread();
  void StopQueue();
  void ReturnError(RunAsyncCallback callback, Status ret, const string &log);

  void ParseInputsDimsForData(const std::vector<ge::Tensor> &input_tensor);
  Status ParseInputsDimsForGetNextNoSinkAndData(const vector<NodePtr> &dynamic_nodes,
                                               const std::vector<ge::Tensor> &input_tensor);
  Status ParseInputsDims(const std::vector<ge::Tensor> &input_tensor);

  bool init_flag_{false};
  bool train_graph_flag_{false};
  uint64_t session_id_{0};
  GraphExecutor graph_executor_;

  std::mutex mutex_;
  std::map<GraphId, GraphNodePtr> graph_nodes_;

  std::thread run_thread_;
  std::atomic_bool thread_run_flag_{false};
  BlockingQueue<RunArgs> run_args_q_;

  // for run graph synchronous return
  std::mutex sync_run_mutex_;
  std::condition_variable condition_;
  // run graph synchronization call back listener
  std::shared_ptr<GraphModelListener> graph_run_listener_;
};
}
#endif // GE_GRAPH_EXECUTE_MODEL_EXECUTOR_H