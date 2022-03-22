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

#ifndef GE_GRAPH_OPTIMIZE_GRAPH_OPTIMIZE_H_
#define GE_GRAPH_OPTIMIZE_GRAPH_OPTIMIZE_H_

#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "common/optimizer/graph_optimizer.h"
#include "graph/compute_graph.h"
#include "graph/manager/graph_context.h"
#include "graph/manager/graph_manager_utils.h"
#include "framework/omg/omg_inner_types.h"

namespace ge {
using ComputeGraphPtr = std::shared_ptr<ge::ComputeGraph>;
using GraphOptimizerPtr = std::shared_ptr<ge::GraphOptimizer>;
class GraphOptimize {
 public:
  GraphOptimize();
  ~GraphOptimize() = default;

  // subgraph optimize
  Status OptimizeSubGraph(ComputeGraphPtr &compute_graph, const std::string &engine_name);

  // original graph optimize
  Status OptimizeOriginalGraph(ComputeGraphPtr &compute_graph);

  Status OptimizeOriginalGraphJudgeInsert(ComputeGraphPtr &compute_graph);

  // for fe prepare optimize in quantize scene
  Status OptimizeOriginalGraphForQuantize(ComputeGraphPtr &compute_graph);

  // for engine to optimize merged whole graph before ge Optimize2
  Status OptimizeWholeGraph(ComputeGraphPtr &compute_graph);

  // for rts optimize before build to add attr and insert memcpy op
  Status OptimizeGraphBeforeBuildForRts(ComputeGraphPtr &compute_graph);

  // optimize whole graph, using after stage1
  Status OptimizeAfterStage1(ComputeGraphPtr &graph);

  // set options
  Status SetOptions(const GraphManagerOptions &options);

  const std::map<uint32_t, std::map<string, size_t>> &GetSummaryOutputIndexes() const {
    return summary_output_indexes_;
  }  // lint !e1073

  // handle summary node before preRun graph
  Status HandleSummaryOp(ComputeGraphPtr &compute_graph);

  // Identify reference node before optimize subgraph
  Status IdentifyReference(ComputeGraphPtr &compute_graph);

  Status HandleMemoryRWConflict(ComputeGraphPtr &compute_graph);

  Status CheckRWConflict(ComputeGraphPtr &compute_graph, bool &has_conflict);

  void TranFrameOp(ComputeGraphPtr &compute_graph);

 private:
  std::mutex mutex_;
  domi::FrameworkType optimize_type_;
  std::string cal_config_;
  std::string insert_op_config_;
  std::string core_type_;
  bool train_graph_flag_ = false;
  bool local_fmk_op_flag_ = false;
  // record the summary names for filter sumarry result.
  std::map<uint32_t, std::map<string, size_t>> summary_output_indexes_ = {};
  std::string func_bin_path_;
  std::string build_mode_;
  std::string build_step_;
};
}  // namespace ge
#endif  // GE_GRAPH_OPTIMIZE_GRAPH_OPTIMIZE_H_
