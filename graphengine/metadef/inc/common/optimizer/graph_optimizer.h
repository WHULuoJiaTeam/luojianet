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

#ifndef INC_COMMON_OPTIMIZER_GRAPH_OPTIMIZER_H_
#define INC_COMMON_OPTIMIZER_GRAPH_OPTIMIZER_H_

#include <map>
#include <string>
#include "graph_optimizer_types.h"
#include "optimize_utility.h"
#include "common/ge_inner_error_codes.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph/compute_graph.h"

using std::map;
using std::string;

/*lint -e148*/
namespace ge {
class GraphOptimizer {
 public:
  virtual ~GraphOptimizer() {}

  // initialize graphOptimizer
  virtual Status Initialize(const std::map<std::string, std::string> &options,
                            OptimizeUtility *const optimize_utility) = 0;

  // close graphOptimizer
  virtual Status Finalize() = 0;

  // optimize original graph for FE quant optimize
  virtual Status OptimizeGraphPrepare(ComputeGraph& graph) {
    return SUCCESS;
  }

  // optimize graph before build for RTS
  virtual Status OptimizeGraphBeforeBuild(ComputeGraph& graph) {
    return SUCCESS;
  }

  // optimize original graph, using in graph preparation stage
  virtual Status OptimizeOriginalGraph(ComputeGraph &graph) = 0;

  // optimize original graph, using for conversion operator insert in graph preparation stage
  virtual Status OptimizeOriginalGraphJudgeInsert(ComputeGraph &graph) {
    return SUCCESS;
  }

  // optimize fused graph
  virtual Status OptimizeFusedGraph(ComputeGraph &graph) = 0;

  // optimize whole graph, using after graph merged stage
  virtual Status OptimizeWholeGraph(ComputeGraph &graph) = 0;

  // get attribute of graph optimizer
  virtual Status GetAttributes(GraphOptimizerAttribute &attrs) const = 0;

  // optimize streamed Graph
  virtual Status OptimizeStreamGraph(ComputeGraph &graph, const RunContext &context) { return SUCCESS; }

  // op compile
  virtual Status OptimizeFusedGraphAfterGraphSlice(ComputeGraph &graph) { return SUCCESS; }

  // optimize whole graph, using after stage1
  virtual Status OptimizeAfterStage1(ComputeGraph &graph) { return SUCCESS; }
};
}  // namespace ge
/*lint +e148*/
#endif  // INC_COMMON_OPTIMIZER_GRAPH_OPTIMIZER_H_
