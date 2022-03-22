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

#ifndef COMMON_GRAPH_FORMAT_REFINER_H_
#define COMMON_GRAPH_FORMAT_REFINER_H_

#if defined(_MSC_VER)
#define METADEF_FUNC_VISIBILITY
#else
#define METADEF_FUNC_VISIBILITY __attribute__((visibility("hidden")))
#endif

#include <deque>
#include <string>
#include <unordered_map>
#include <vector>
#include "graph/compute_graph.h"
#include "graph/types.h"
#include "graph/ge_error_codes.h"

namespace ge {
// ShapeRefiner performs shape inference for compute graphs
class METADEF_FUNC_VISIBILITY FormatRefiner {
 public:
  static graphStatus InferOrigineFormat(const ge::ComputeGraphPtr &graph);

 private:
  static graphStatus RefreshConstantOutProcess(const ComputeGraphPtr &com_graph, const OpDescPtr &op_desc);
  static graphStatus GetAnchorPoints(const ge::ComputeGraphPtr &com_graph, std::vector<ge::NodePtr> &anchor_points,
                                     std::vector<ge::NodePtr> &anchor_data_nodes);
  static graphStatus AnchorProcess(const ge::NodePtr &anchor_node);
  static void RefreshOriginFormatOfAnchor(const std::vector<ge::NodePtr> &anchor_points);
  static graphStatus BackInferProcess(std::deque<ge::NodePtr> &nodes, const ge::NodePtr &node);
  static graphStatus ForwardInferProcess(std::deque<ge::NodePtr> &nodes, const ge::NodePtr &node);
  static graphStatus DataNodeFormatProcess(const ComputeGraphPtr &graph,
                                           const std::vector<ge::NodePtr> &anchor_data_nodes,
                                           const ge::Format data_format);
  static bool IsGraphInferred(const ComputeGraphPtr &graph);
};
}  // namespace ge
#endif  // COMMON_GRAPH_FORMAT_REFINER_H_
