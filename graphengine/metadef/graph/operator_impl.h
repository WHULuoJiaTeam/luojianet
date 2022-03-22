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

#ifndef METADEF_CXX_OPERATOR_IMPL_H
#define METADEF_CXX_OPERATOR_IMPL_H
#include <memory>
#include <string>
#include "graph/op_desc.h"
#include "graph/node.h"
#include "graph/operator.h"
#include "graph/runtime_inference_context.h"
#include "op_io.h"
namespace ge {
class OperatorImpl : public std::enable_shared_from_this<OperatorImpl> {
 public:
  explicit OperatorImpl(const std::string &name, const std::string &type);
  explicit OperatorImpl(const OpDescPtr &op_desc);
  explicit OperatorImpl(ConstNodePtr node);
  ~OperatorImpl();

  void SetInputImpl(const std::string &dst_name, const Operator &src_oprt);
  void SetInputImpl(const std::string &dst_name, const OutHandler &out_handler);
  void AddControlInputImp(const Operator &src_oprt);
  graphStatus GetInputImpl(const std::string &dst_name, ge::OpIO &out_handler) const;
  graphStatus GetInputImpl(const uint32_t idx, ge::OpIO &out_handler) const;
  graphStatus GetInputConstData(const std::string &dst_name, Tensor &data);
  graphStatus GetInputConstData(const uint32_t idx, ConstGeTensorPtr &ge_tensor) const;
  graphStatus GetInputConstDataOut(const std::string &dst_name, Tensor &data) const;
  graphStatus GetInputConstDataOut(const uint32_t idx, ConstGeTensorPtr &ge_tensor) const;
  bool InputIsSet(const std::string &name);
  std::string GetName() const;
  GeTensorDesc GetInputDesc(const std::string &name) const;
  GeTensorDesc GetInputDesc(const uint32_t index) const;
  graphStatus UpdateInputDesc(const std::string &name, const GeTensorDesc &tensor_desc);
  OutHandler GetOutput(const std::string &name);
  OutHandler GetOutput(uint32_t index);
  GeTensorDesc GetOutputDesc(const std::string &name) const;
  GeTensorDesc GetOutputDesc(const uint32_t index) const;
  graphStatus UpdateOutputDesc(const std::string &name, const GeTensorDesc &tensor_desc);
  size_t GetInputsSize() const;
  size_t GetOutputsSize() const;
  graphStatus SetAttr(const std::string &name, const AnyValue &&attr_value);
  graphStatus GetAttr(const std::string &name, AnyValue &attr_value) const;
  OpDescPtr GetOpDescImpl() const;
  void UpdateLinkMapImpl(const std::string &src_name, const OpIO &op_dst);
  Operator ToOperator();
  void ClearOutputLinks() noexcept;
  void ClearInputLinks() noexcept;
  ge::ConstNodePtr GetNode() const;
  void SetInferenceContext(const InferenceContextPtr &inference_context);
  InferenceContextPtr GetInferenceContext() const;
  void SubgraphRegister(const std::string &ir_name, const bool dynamic);
  void SubgraphCountRegister(const std::string &ir_name, const uint32_t count);
  void SetSubgraphBuilder(const std::string &ir_name, const uint32_t index, const SubgraphBuilder &builder);
  SubgraphBuilder GetSubgraphBuilder(const std::string &ir_name, const uint32_t index) const;
  SubgraphBuilder GetSubgraphBuilder(const std::string &name) const;
  std::vector<std::string> GetSubgraphNames() const;
  size_t GetSubgraphNamesCount() const;

  static OpDescPtr GetOpDesc(const Operator &oprt);

 public:
  OpDescPtr op_desc_ = nullptr;

 private:
  graphStatus GetFromPeerNode(NodePtr &peer_node, const OutDataAnchorPtr &out_data_anchor,
                              ConstGeTensorPtr &ge_tensor) const;

 private:
  ge::ConstNodePtr node_{nullptr};
  ge::InferenceContextPtr inference_context_;
  std::map<std::string, std::vector<OpIO>> output_links_{};
  std::map<std::string, OpIO> input_link_{};
  std::vector<std::weak_ptr<OperatorImpl>> control_input_link_{};
  std::vector<std::weak_ptr<OperatorImpl>> control_output_link_{};
  std::map<std::string, SubgraphBuilder> subgraph_names_to_builders_;
  RuntimeInferenceContext *runtime_context_{nullptr};

 private:
  friend class GraphBuilderImpl;
  friend class OpDescUtils;
};

}  // namespace ge

#endif  //METADEF_CXX_OPERATOR_IMPL_H
