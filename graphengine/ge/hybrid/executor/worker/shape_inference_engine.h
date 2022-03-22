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

#ifndef GE_HYBRID_EXECUTOR_INFERSHAPE_SHAPE_INFERENCE_ENGINE_H_
#define GE_HYBRID_EXECUTOR_INFERSHAPE_SHAPE_INFERENCE_ENGINE_H_

#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/executor/subgraph_context.h"
#include <mutex>

namespace ge {
namespace hybrid {
class ShapeInferenceEngine {
 public:
  ShapeInferenceEngine(GraphExecutionContext *execution_context, SubgraphContext *subgraph_context);
  ~ShapeInferenceEngine() = default;

  Status InferShape(NodeState &node_state);

  Status InferShapeForSubgraph(const NodeItem &node_item, const FusedSubgraph &fused_subgraph);

  Status PropagateOutputShapes(NodeState &node_state);

  static Status CalcOutputTensorSizes(const NodeItem &node_item, bool fallback_with_range = false);

 private:
  static Status CanonicalizeShape(GeTensorDesc &tensor_desc, std::vector<int64_t> &shape, bool fallback_with_range);
  static Status CalcTensorSize(DataType data_type, const std::vector<int64_t> &shape, int64_t &tensor_size);
  static Status UpdatePeerNodeShape(const Node &node);
  Status AwaitDependentNodes(NodeState &node_state);

  GraphExecutionContext *execution_context_;
  SubgraphContext *subgraph_context_;
  std::mutex mu_;
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_EXECUTOR_INFERSHAPE_SHAPE_INFERENCE_ENGINE_H_
