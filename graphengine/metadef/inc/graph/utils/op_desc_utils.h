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

#ifndef INC_GRAPH_UTILS_OP_DESC_UTILS_H_
#define INC_GRAPH_UTILS_OP_DESC_UTILS_H_

#include <string>
#include <vector>
#include "graph/def_types.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/range_vistor.h"
#include "graph/runtime_inference_context.h"
/*lint -e148*/
namespace ge {
class OpDesc;
using OpDescPtr = std::shared_ptr<OpDesc>;
using ConstGeTensorBarePtr = const GeTensor*;

class OpDescUtils {
 public:
  template <class T>
  using Vistor = RangeVistor<T, std::shared_ptr<OpDesc>>;

  OpDescUtils() = default;
  ~OpDescUtils() = default;
  static bool HasQuantizeFactorParams(const OpDescPtr& op_desc);
  static bool HasQuantizeFactorParams(const OpDesc& op_desc);
  static std::vector<ge::NodePtr> GetConstInputNode(const ge::Node& node);
  static std::vector<NodeToOutAnchor> GetConstInputNodeAndAnchor(const ge::Node &node);
  static std::vector<ConstGeTensorPtr> GetInputData(const std::vector<ge::NodePtr>& input_nodes);
  static std::vector<ConstGeTensorPtr> GetWeightsFromNodes(const std::vector<NodeToOutAnchor>& input_nodes_2_out_anchors);

  static std::vector<ConstGeTensorPtr> GetWeights(const ge::Node& node);
  static std::vector<ConstGeTensorPtr> GetWeights(const ge::ConstNodePtr& node);
  static std::vector<GeTensorPtr> MutableWeights(const ge::Node& node);
  static std::vector<GeTensorPtr> MutableWeights(const ge::NodePtr node);
  static graphStatus SetWeights(ge::Node& node, const std::vector<ge::GeTensorPtr>& weights);
  static graphStatus SetWeights(ge::NodePtr node, const std::vector<ge::GeTensorPtr> &weights);
  static graphStatus SetWeights(ge::Node &node, const std::map<int, ge::GeTensorPtr> &weights_map);
  static graphStatus ClearWeights(const ge::NodePtr node);

  static bool ClearInputDesc(const ge::OpDescPtr op_desc, const uint32_t index);
  static bool ClearInputDesc(const ge::NodePtr& node);
  static bool ClearOutputDesc(const ge::OpDescPtr& op_desc, const uint32_t index);
  static bool ClearOutputDesc(const ge::NodePtr& node);
  static std::vector<ge::NodePtr> GetConstInputs(const ge::Node& node);
  static std::vector<ge::NodePtr> GetConstInputs(const ge::ConstNodePtr& node);
  static size_t GetNonConstInputsSize(const ge::Node& node);
  static size_t GetNonConstInputsSize(const ge::ConstNodePtr node);
  // Index: Indicates the index of all non const inputs
  static GeTensorDesc GetNonConstInputTensorDesc(const ge::Node& node, const size_t index_non_const = 0);
  static GeTensorDesc GetNonConstInputTensorDesc(const ge::ConstNodePtr& node, const size_t index_non_const = 0);
  static bool GetNonConstInputIndex(const ge::Node& node, const size_t index_non_const, size_t& index);
  static bool GetNonConstInputIndex(const ge::ConstNodePtr& node, const size_t index_non_const, size_t& index);
  // Index: Indicates the index of all inputs
  static bool IsNonConstInput(const ge::Node& node, const size_t index = 0);
  static bool IsNonConstInput(const ge::ConstNodePtr& node, const size_t index = 0);

  static std::vector<ge::GeTensorDesc> GetNonConstTensorDesc(const ge::ConstNodePtr& node);
  static graphStatus AddConstOpToAnchor(const InDataAnchorPtr in_anchor, const GeTensorPtr& tensor_ptr);

  static Operator CreateOperatorFromOpDesc(OpDescPtr op_desc);
  static Operator CreateOperatorFromNode(ge::ConstNodePtr node_ptr);
  static OpDescPtr GetOpDescFromOperator(const Operator& oprt);
  static graphStatus CopyOperatorLinks(const std::map<std::string, ge::Operator> &src_op_list,
                                       std::map<std::string, ge::Operator> &dst_op_list);
  static graphStatus CopyOperators(const ComputeGraphPtr &dst_compute_graph,
                                   const std::map<ConstNodePtr, NodePtr> &node_old_2_new,
                                   const std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new,
                                   const std::map<std::string, ge::Operator> &src_op_list,
                                   std::map<std::string, ge::Operator> &dst_op_list);
  static OpDescPtr CreateConstOp(const GeTensorPtr& tensor_ptr);

  static graphStatus SetSubgraphInstanceName(const std::string &subgraph_name,
      const std::string &subgraph_instance_name, OpDescPtr &op_desc);
  static ConstGeTensorBarePtr GetInputConstData(const Operator &op, uint32_t idx);
  static void SetRuntimeContextToOperator(const Operator &op, RuntimeInferenceContext *context);
 private:
  static GeTensorPtr MutableWeights(ge::OpDesc& op_desc);
  static GeTensorPtr MutableWeights(const ge::OpDescPtr op_desc);
  static graphStatus SetWeights(ge::OpDesc& op_desc, const GeTensorPtr weight);
  static graphStatus SetWeights(ge::OpDescPtr op_desc, const GeTensorPtr weight);
};

class OpDescBuilder {
 public:
  OpDescBuilder(std::string name, std::string type) : name_(std::move(name)), type_(std::move(type)) {}
  OpDescBuilder(const OpDescBuilder &) = delete;
  OpDescBuilder &operator=(const OpDescBuilder &) = delete;
  OpDescBuilder(const OpDescBuilder &&) = delete;
  OpDescBuilder &operator=(const OpDescBuilder &&) = delete;
  ~OpDescBuilder() = default;

  ///
  /// @brief Add input
  /// @param [in] name
  /// @return OpDescBuilder
  ///
  OpDescBuilder& AddInput(const std::string &name);

  ///
  /// @brief Add input
  /// @param [in] name
  /// @param [in] tensor
  /// @return OpDescBuilder
  ///
  OpDescBuilder& AddInput(const std::string &name, const GeTensorDesc &tensor);

  ///
  /// @brief Add dynamic input
  /// @param [in] name
  /// @param [in] num
  /// @return OpDescBuilder
  ///
  OpDescBuilder& AddDynamicInput(const std::string &name, const uint32_t num);

  ///
  /// @brief Add dynamic input
  /// @param [in] name
  /// @param [in] num
  /// @param [in] tensor
  /// @return OpDescBuilder
  ///
  OpDescBuilder& AddDynamicInput(const std::string &name, const uint32_t num, const GeTensorDesc &tensor);

  ///
  /// @brief Add output
  /// @param [in] name
  /// @return OpDescBuilder
  ///
  OpDescBuilder& AddOutput(const std::string &name);

  ///
  /// @brief Add output
  /// @param [in] name
  /// @param [in] tensor
  /// @return OpDescBuilder
  ///
  OpDescBuilder& AddOutput(const std::string &name, const GeTensorDesc &tensor);

  ///
  /// @brief Add dynamic output
  /// @param [in] name
  /// @param [in] num
  /// @return OpDescBuilder
  ///
  OpDescBuilder& AddDynamicOutput(const std::string &name, const uint32_t num);

  ///
  /// @brief Add dynamic output
  /// @param [in] name
  /// @param [in] num
  /// @param [in] tensor
  /// @return OpDescBuilder
  ///
  OpDescBuilder& AddDynamicOutput(const std::string &name, const uint32_t num, const GeTensorDesc &tensor);

  ///
  /// @brief Build op_desc
  /// @return OpDescPtr
  ///
  OpDescPtr Build();

 private:
  std::string name_;
  std::string type_;
  std::vector<std::pair<std::string, GeTensorDesc>> inputs_;
  std::vector<std::pair<std::string, GeTensorDesc>> outputs_;
};
}  // namespace ge
/*lint +e148*/
#endif  // INC_GRAPH_UTILS_OP_DESC_UTILS_H_
