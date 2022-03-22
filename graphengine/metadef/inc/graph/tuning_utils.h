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
#ifndef MAIN_TUNING_UTILS_H
#define MAIN_TUNING_UTILS_H

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <queue>
#include <mutex>

#include <graph/anchor.h>
#include <graph/detail/attributes_holder.h>
#include <graph/ge_tensor.h>
#include <graph/graph.h>
#include <graph/model.h>
#include <graph/node.h>
#include <graph/utils/graph_utils.h>
#include <graph/utils/type_utils.h>

#include "framework/common/debug/ge_log.h"
#include "utils/attr_utils.h"
#include "utils/node_utils.h"
#include "external/ge/ge_api_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
namespace ge {
// Configure build mode, default value is "normal"
constexpr char_t BUILD_MODE[] = "ge.buildMode";
constexpr char_t BUILD_STEP[] = "ge.buildStep";
// Configure tuning path
constexpr char_t TUNING_PATH[] = "ge.tuningPath";
// for interface: aclgrphBuildModel
extern const std::set<std::string> ir_builder_supported_options_for_lx_fusion;

// Build model
constexpr char_t BUILD_MODE_NORMAL[] = "normal";
constexpr char_t BUILD_MODE_TUNING[] = "tuning";
constexpr char_t BUILD_MODE_BASELINE[] = "baseline";
extern const std::set<std::string> build_mode_options;

// Build step
constexpr char_t BUILD_STEP_BEFORE_UB_MATCH[] = "before_ub_match";
constexpr char_t BUILD_STEP_AFTER_UB_MATCH[] = "after_ub_match";
constexpr char_t BUILD_STEP_AFTER_BUILDER[] = "after_builder";
constexpr char_t BUILD_STEP_AFTER_BUILDER_SUB[] = "after_builder_sub";
constexpr char_t BUILD_STEP_AFTER_MERGE[] = "after_merge";
extern const std::set<std::string> build_step_options;

using SubgraphCreateOutNode = std::unordered_map<ComputeGraphPtr, NodePtr>;
using NodetoNodeMap = std::unordered_map<NodePtr, NodePtr>;
using NodeVec = std::vector<NodePtr>;
using NodeNametoNodeNameMap = std::map<std::string, std::string>;
using NodetoNodeNameMap = std::unordered_map<NodePtr, std::string>;
class TuningUtils {
 public:
  TuningUtils() = default;
  ~TuningUtils() = default;
  // Dump all the subgraphs and modify
  // the subgraphs in them to be executable subgraphs if exe_flag is true
  // `tuning_path` means path to save the graphs
  static graphStatus ConvertGraphToFile(std::vector<ComputeGraphPtr> tuning_subgraphs,
                                        std::vector<ComputeGraphPtr> non_tuning_subgraphs = {},
                                        const bool exe_flag = false,
                                        const std::string &path = "",
                                        const std::string &user_path = "");
  // Recovery `graph` from graph dump files configured in options
  static graphStatus ConvertFileToGraph(const std::map<int64_t, std::string> &options, ge::Graph &graph);

private:
  // part 1
  class HelpInfo {
    HelpInfo(const int64_t index, const bool exe_flag, const bool is_tuning_graph, const std::string &path,
             const std::string &user_path) : index_(index),
                                             exe_flag_(exe_flag),
                                             is_tuning_graph_(is_tuning_graph),
                                             path_(path),
                                             user_path_(user_path) {}
    ~HelpInfo() = default;
   private:
    int64_t index_;
    bool exe_flag_;
    bool is_tuning_graph_;
    const std::string &path_;
    const std::string &user_path_;
    friend class TuningUtils;
  };
  static graphStatus MakeExeGraph(ComputeGraphPtr &exe_graph,
                                  const HelpInfo& help_info);
  static graphStatus ConvertConstToWeightAttr(const ComputeGraphPtr &exe_graph);
  static graphStatus HandlePld(NodePtr &node);
  static graphStatus HandleEnd(NodePtr &node);
  static graphStatus ChangePld2Data(const NodePtr &node, const NodePtr &data_node);
  static graphStatus ChangeEnd2NetOutput(NodePtr &end_node, NodePtr &out_node);
  static graphStatus LinkEnd2NetOutput(NodePtr &end_node, NodePtr &out_node);
  static graphStatus CreateDataNode(NodePtr &node, NodePtr &data_node);
  static graphStatus CreateNetOutput(const NodePtr &node, NodePtr &out_node);
  static graphStatus AddAttrToDataNodeForMergeGraph(const NodePtr &pld, const NodePtr &data_node);
  static graphStatus AddAttrToNetOutputForMergeGraph(const NodePtr &end, const NodePtr &out_node, const int64_t index);
  static void DumpGraphToPath(const ComputeGraphPtr &exe_graph, const int64_t index,
                              const bool is_tuning_graph, std::string path);

  static SubgraphCreateOutNode create_output_;
  // part 2
  static graphStatus MergeAllSubGraph(std::vector<ComputeGraphPtr> &subgraphs,
                                      ComputeGraphPtr &output_merged_compute_graph);
  static graphStatus MergeSubGraph(const ComputeGraphPtr &subgraph);
  // Deletes new data and output nodes added by call `MakeExeGraph()` func in part 1
  static graphStatus RemoveDataNetoutputEdge(ComputeGraphPtr &graph);
  static graphStatus HandleContinuousInputNodeNextData(const NodePtr &node);
  static NodePtr FindNode(const std::string &name, int64_t &in_index);

  static NodeNametoNodeNameMap data_2_end_;
  static NodetoNodeNameMap data_node_2_end_node_;
  static NodetoNodeMap data_node_2_netoutput_node_;
  static NodeVec netoutput_nodes_;
  static NodeVec merged_graph_nodes_;
  static std::mutex mutex_;
  // for debug
  static std::string PrintCheckLog();
  static std::string GetNodeNameByAnchor(const Anchor * const anchor);
};
}
#endif //MAIN_TUNING_UTILS_H
