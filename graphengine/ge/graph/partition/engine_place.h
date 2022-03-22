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

#ifndef GE_GRAPH_PARTITION_ENGINE_PLACE_H_
#define GE_GRAPH_PARTITION_ENGINE_PLACE_H_

#include <string>
#include <unordered_map>

#include "framework/common/ge_inner_error_codes.h"
#include "graph/compute_graph.h"

namespace ge {
using NodeEngineMap = std::unordered_map<ConstNodePtr, std::string>;

///
/// @ingroup graph/partition
/// @brief Assigned individual DNNEngine to each node in the origin graph
/// @author
///
class EnginePlacer {
 public:
  explicit EnginePlacer(const ComputeGraphPtr &graph) : compute_graph_(graph) {}
  EnginePlacer() = default;
  ~EnginePlacer() = default;

  Status Run();

  // Get the unique node-engine map
  const NodeEngineMap *GetNodeEngineMap() const { return &node_engine_map_; }

  void SetComputeGraph(const ComputeGraphPtr &compute_graph) { compute_graph_ = compute_graph; }

 private:
  Status AssignEngineAndLog(ConstNodePtr node_ptr, const std::string &engine_name);
  Status Check() const;

  ComputeGraphPtr compute_graph_;
  NodeEngineMap node_engine_map_;
};
}  // namespace ge

#endif  // GE_GRAPH_PARTITION_ENGINE_PLACE_H_
