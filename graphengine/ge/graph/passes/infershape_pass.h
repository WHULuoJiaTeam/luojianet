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

#ifndef GE_GRAPH_PASSES_INFERSHAPE_PASS_H_
#define GE_GRAPH_PASSES_INFERSHAPE_PASS_H_

#include "graph/passes/infer_base_pass.h"
#include <stack>

namespace ge {
class InferShapePass : public InferBasePass {
 public:
  std::string SerialTensorInfo(const GeTensorDescPtr &tensor_desc) const override;
  graphStatus Infer(NodePtr &node) override;

  graphStatus UpdateTensorDesc(const GeTensorDescPtr &src, GeTensorDescPtr &dst, bool &changed) override;
  graphStatus UpdateOutputFromSubgraphs(const std::vector<GeTensorDescPtr> &src, GeTensorDescPtr &dst) override;
  graphStatus UpdateOutputFromSubgraphsForMultiDims(const std::vector<GeTensorDescPtr> &src,
                                                            GeTensorDescPtr &dst) override;

  Status OnSuspendNodesLeaked() override;

 private:
  graphStatus InferShapeAndType(NodePtr &node);
  graphStatus CallInferShapeFunc(NodePtr &node, Operator &op);
  bool SameTensorDesc(const GeTensorDescPtr &src, const GeTensorDescPtr &dst);
  void UpdateCurNodeOutputDesc(NodePtr &node);
  Status SuspendV1LoopExitNodes(const NodePtr &node);
  struct SuspendNodes {
    std::stack<NodePtr> nodes;
    std::unordered_set<NodePtr> nodes_set;

    NodePtr PopSuspendedNode() {
      auto top_node = nodes.top();
      nodes.pop();
      nodes_set.erase(top_node);
      return top_node;
    }
  };
  std::map<std::string, SuspendNodes> graphs_2_suspend_nodes_;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_INFERSHAPE_PASS_H_
