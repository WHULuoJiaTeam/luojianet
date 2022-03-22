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

#ifndef INC_GRAPH_SHAPE_REFINER_H_
#define INC_GRAPH_SHAPE_REFINER_H_

#include <string>
#include "external/graph/inference_context.h"

#include "external/graph/ge_error_codes.h"
#include "graph/node.h"
#include "graph/resource_context_mgr.h"

namespace ge {
// ShapeRefiner performs shape inference for compute graphs
class ShapeRefiner {
 public:
  static graphStatus InferShapeAndType(const ConstNodePtr &node, Operator &op, const bool before_subgraph);
  static graphStatus InferShapeAndType(const NodePtr &node, const bool before_subgraph);
  static graphStatus InferShapeAndType(const NodePtr &node);
  static graphStatus InferShapeAndType(const ConstNodePtr &node, Operator &op);
  static graphStatus DoInferShapeAndTypeForRunning(const ConstNodePtr &node, Operator &op, const bool before_subgraph);
  static graphStatus InferShapeAndTypeForRunning(const NodePtr &node, Operator &op, const bool before_subgraph);
  static void ClearContextMap();
  static graphStatus CreateInferenceContext(const NodePtr &node,
                                            InferenceContextPtr &inference_context);
  static graphStatus CreateInferenceContext(const NodePtr &node,
                                            ResourceContextMgr *resource_context_mgr,
                                            InferenceContextPtr &inference_context);
  static void PushToContextMap(const NodePtr &node, const InferenceContextPtr &inference_context);

 private:
  static void PrintInOutTensorShape(const ge::NodePtr &node, const std::string &phase);
  static graphStatus GetRealInNodesAndIndex(NodePtr &input_node, int32_t &output_idx,
                                            std::map<NodePtr, int32_t> &nodes_idx);
  static graphStatus PostProcessAfterInfershape(const NodePtr &node, const Operator &op, const bool is_unknown_graph);
  static graphStatus UpdateInputOutputDesc(const NodePtr &node);
};
}  // namespace ge
#endif  // INC_GRAPH_SHAPE_REFINER_H_
