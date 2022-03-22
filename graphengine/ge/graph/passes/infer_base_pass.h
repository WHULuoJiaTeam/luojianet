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
#ifndef GE_GRAPH_PASSES_INFER_BASE_PASS_H_
#define GE_GRAPH_PASSES_INFER_BASE_PASS_H_

#include "graph/passes/base_pass.h"

namespace ge {
class InferBasePass : public BaseNodePass {
 public:
  Status Run(NodePtr &node) override;
  graphStatus InferAndUpdate(NodePtr &node, bool before_subgraph, std::set<NodePtr> &changed_nodes);
  void PrintInOutTensors(const NodePtr &node, const std::string &phase);

 protected:
  virtual std::string SerialTensorInfo(const GeTensorDescPtr &tensor_desc) const = 0;
  virtual bool NeedInfer(const NodePtr &node) const;
  virtual graphStatus Infer(NodePtr &node) = 0;

  /**
   * Update the output TensorDesc by src TensorDesc. This will be called when updating peer node input desc.
   * @param src, input TensorDesc
   * @param dst, output TensorDesc to be updated
   * @return
   */
  virtual graphStatus UpdateTensorDesc(const GeTensorDescPtr &src, GeTensorDescPtr &dst, bool &changed) = 0;

  /**
   * Update the output TensorDesc for nodes which contain subgraphs.
   * In dynamic multi-dims/batch/images size scene, the update process maybe different,
   * in which case, the `InferBasePass` will call method `UpdateOutputFromSubgraphsForMultiDims` instead.
   * @param src, input TensorDesc from NetOutput nodes in all subgraphs
   * @param dst, output TensorDesc to be updated
   * @return
   */
  virtual graphStatus UpdateOutputFromSubgraphs(const std::vector<GeTensorDescPtr> &src,
                                                GeTensorDescPtr &dst) = 0;
  virtual graphStatus UpdateOutputFromSubgraphsForMultiDims(const std::vector<GeTensorDescPtr> &src,
                                                            GeTensorDescPtr &dst) = 0;

 private:
  void AddChangedNodesImmediateRepass(const std::set<NodePtr> &changed_nodes);
  bool ContainsSubgraph(const NodePtr &node);
  std::vector<ComputeGraphPtr> GetCurNodeSubgraphs(const NodePtr &node);
  graphStatus UpdateTensorDescToSubgraphData(NodePtr &node);
  graphStatus UpdateTensorDescToParentNodeOutput(NodePtr &node);
  graphStatus UpdateParentNodeContainsSubgraphs(NodePtr &node,
                                                const std::vector<std::vector<GeTensorDescPtr>> &ref_out_tensors);
  graphStatus UpdateTensorDescToPeerInputs(NodePtr &node, std::set<NodePtr> &changed_nodes);
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_INFER_BASE_PASS_H_
