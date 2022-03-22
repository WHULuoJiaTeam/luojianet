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

#ifndef GE_GRAPH_PARTITION_STAGE_PARTITION_H_
#define GE_GRAPH_PARTITION_STAGE_PARTITION_H_

#include <map>
#include <unordered_set>
#include <list>
#include <utility>
#include "framework/common/ge_inner_error_codes.h"
#include "graph/compute_graph.h"

namespace ge {
struct StageInfo {
  explicit StageInfo(uint32_t level) : stage_level(level) {}
  uint32_t stage_level;
  std::unordered_set<NodePtr> stage_nodes;
  std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> data_inputs;
  std::vector<std::pair<OutDataAnchorPtr, std::list<InDataAnchorPtr>>> data_outputs;
  std::list<std::pair<OutControlAnchorPtr, InControlAnchorPtr>> ctrl_inputs;
  std::list<std::pair<OutControlAnchorPtr, InControlAnchorPtr>> ctrl_outputs;
  std::list<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> inner_data_edges;
  std::list<std::pair<OutControlAnchorPtr, InControlAnchorPtr>> inner_ctrl_edges;
};

class StagePartitioner {
 public:
  explicit StagePartitioner(ComputeGraphPtr graph) : root_graph_(std::move(graph)) {}
  ~StagePartitioner() = default;

  Status Partition();

 private:
  Status SplitStageLevel();

  Status StagePartition();

  static void FindStageIO(const std::unordered_set<NodePtr> &stage_nodes, StageInfo &stage_info);

  NodePtr BuildSubgraphNode(const std::string &graph_name, const StageInfo &stage_info);

  static ComputeGraphPtr BuildStageGraph(const NodePtr &subgraph_node, const StageInfo &stage_info);

  static Status RelinkDataEdges(const NodePtr &subgraph_node, const StageInfo &stage_info);

  static Status RelinkCtrlEdges(const NodePtr &subgraph_node, const StageInfo &stage_info);

  ComputeGraphPtr root_graph_;
  std::map<uint32_t, std::unordered_set<NodePtr>> stage_nodes_;
};
}  // namespace ge

#endif  // GE_GRAPH_PARTITION_STAGE_PARTITION_H_
