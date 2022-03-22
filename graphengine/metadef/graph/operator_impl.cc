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

#include "operator_impl.h"

#include "op_io.h"
#include "debug/ge_log.h"
#include "debug/ge_util.h"
#include "debug/ge_op_types.h"
#include "graph/compute_graph.h"
#include "graph/ge_context.h"
#include "graph/runtime_inference_context.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/constant_utils.h"
#include "graph/utils/node_utils.h"

namespace ge {
OperatorImpl::OperatorImpl(const std::string &name, const std::string &type)
    : enable_shared_from_this(), op_desc_(ComGraphMakeShared<OpDesc>(name, type)) {
  if (op_desc_ == nullptr) {
    GELOGW("[Check][Param] Make op_desc failed");
  }
}

OperatorImpl::OperatorImpl(const OpDescPtr &op_desc) : enable_shared_from_this(), op_desc_(op_desc) {}

OperatorImpl::OperatorImpl(ge::ConstNodePtr node) : node_(std::move(node)) {
  if (node_ != nullptr && node_->GetOpDesc() != nullptr) {
    op_desc_ = node_->GetOpDesc();
  }
}

OperatorImpl::~OperatorImpl() {}

void OperatorImpl::SetInputImpl(const std::string &dst_name, const ge::Operator &src_oprt) {
  GE_CHK_BOOL_EXEC(src_oprt.GetOutputsSize() == 1U,
                   REPORT_INNER_ERROR("E19999", "The source operator[%s] must be single output operator",
                                      src_oprt.operator_impl_->op_desc_->GetName().c_str());
                   return, "[Check][Param] The source operator[%s] must be single output operator",
                         src_oprt.operator_impl_->op_desc_->GetName().c_str());

  const auto out_handler = src_oprt.GetOutput(0U);
  if (out_handler == nullptr) {
    return;
  }

  return SetInputImpl(dst_name, out_handler);
}

void OperatorImpl::SetInputImpl(const std::string &dst_name, const ge::OutHandler &out_handler) {
  GE_CHK_BOOL_EXEC(out_handler != nullptr, REPORT_INNER_ERROR("E19999", "param out_handler is nullptr, check invalid.");
                   return, "[Check][Param] SetInputImpl faild, as out_handler is nullptr.");
  GE_CHK_BOOL_EXEC(!dst_name.empty(), REPORT_INNER_ERROR("E19999", "param dst_name is empty, check invalid.");
                   return, "[Check][Param] dst name is empty");
  GE_CHK_BOOL_EXEC(op_desc_ != nullptr, REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr.");
                   return, "[Check][Param] op_desc_ is nullptr.");
  (void)input_link_.insert(std::make_pair(dst_name, *out_handler));

  const std::string src_name = out_handler->GetName();
  const int32_t dst_index = op_desc_->GetInputIndexByName(dst_name);
  GE_CHK_BOOL_EXEC(dst_index >= 0,
                   REPORT_INNER_ERROR("E19999", "Find input index by name failed. name[%s], op name:%s",
                                      dst_name.c_str(), op_desc_->GetName().c_str());
                   return, "[Get][InputIndex] Find input index by name failed. name[%s], op name:%s", dst_name.c_str(),
                         op_desc_->GetName().c_str());
  const auto out_op_impl = out_handler->GetOwner();
  GE_CHK_BOOL_EXEC(out_op_impl != nullptr && out_op_impl->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "out_handler invalid. name[%s]", dst_name.c_str());
                   return, "[Get][Impl] out_handler invalid. name[%s]", dst_name.c_str());
  bool is_const = false;
  if (out_op_impl->GetOpDescImpl()->GetType() == CONSTANT) {
    is_const = true;
  }
  auto is_input_const = op_desc_->GetIsInputConst();
  for (int32_t i = static_cast<int32_t>(is_input_const.size()); i <= dst_index; ++i) {
    is_input_const.push_back(false);
  }
  is_input_const[static_cast<size_t>(dst_index)] = is_const;
  op_desc_->SetIsInputConst(is_input_const);

  OpIO in_handler(dst_name, dst_index, shared_from_this());
  GE_CHK_BOOL_EXEC(out_op_impl != nullptr,
                   REPORT_INNER_ERROR("E19999", "out_handler invalid. name[%s]", dst_name.c_str());
                   return, "[Get][Impl] of out_handler failed.");

  out_op_impl->UpdateLinkMapImpl(src_name, in_handler);
  auto src_output_desc = out_op_impl->GetOutputDesc(src_name);
  const auto dst_input_desc = op_desc_->GetInputDesc(dst_name);
  if (dst_input_desc.GetFormat() == FORMAT_RESERVED) {
    src_output_desc.SetFormat(FORMAT_ND);
    src_output_desc.SetOriginFormat(FORMAT_ND);
  } else {
    src_output_desc.SetFormat(dst_input_desc.GetFormat());
    src_output_desc.SetOriginFormat(dst_input_desc.GetOriginFormat());
  }
  GE_CHK_BOOL_EXEC(op_desc_->UpdateInputDesc(dst_name, src_output_desc) == GRAPH_SUCCESS,
                   REPORT_CALL_ERROR("E19999", "UpdateInputDesc failed, dst name is %s, src name is %s",
                                     dst_name.c_str(), src_name.c_str());
                   return, "[Update][InputDesc] failed, dst name is %s, src name is %s", dst_name.c_str(),
                         src_name.c_str());  // fix for linking opdesc
}

void OperatorImpl::AddControlInputImp(const ge::Operator &src_oprt) {
  if (src_oprt.operator_impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "Src operator impl is nullptr, check invalid");
    GELOGE(FAILED, "[Check][Param] Src operator impl is nullptr");
    return;
  }
  for (auto &input : control_input_link_) {
    if (input.lock() == src_oprt.operator_impl_) {
      return;
    }
  }
  control_input_link_.push_back(src_oprt.operator_impl_);
  src_oprt.operator_impl_->control_output_link_.push_back(shared_from_this());
}

graphStatus OperatorImpl::GetInputImpl(const std::string &dst_name, ge::OpIO &out_handler) const {
  const auto out = input_link_.find(dst_name);
  if (out == input_link_.end()) {
    return GRAPH_FAILED;
  }
  out_handler = out->second;
  return GRAPH_SUCCESS;
}

graphStatus OperatorImpl::GetInputImpl(const uint32_t idx, ge::OpIO &out_handler) const {
  GE_CHECK_NOTNULL(op_desc_);
  const std::string dst_name = op_desc_->GetInputNameByIndex(idx);
  return GetInputImpl(dst_name, out_handler);
}

namespace {
graphStatus GetFromInputDesc(const OpDescPtr &op_desc, const int32_t index, ConstGeTensorPtr &ge_tensor) {
  // if tensor has host mem, init data by ATTR_NAME_VALUE first
  const auto tensor = op_desc->MutableInputDesc(static_cast<uint32_t>(index));
  GeTensorPtr tensor_value = nullptr;
  if (AttrUtils::MutableTensor(tensor, ATTR_NAME_VALUE, tensor_value)) {
    GELOGD("Get ATTR_NAME_VALUE from %d input of %s, Tensor addr is %p, tensor value data type is %d.", index,
           op_desc->GetName().c_str(), tensor.get(), tensor_value->GetTensorDesc().GetDataType());
    ge_tensor = tensor_value;
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
}  // namespace


graphStatus OperatorImpl::GetFromPeerNode(NodePtr &peer_node,
                                          const OutDataAnchorPtr &out_data_anchor,
                                          ConstGeTensorPtr &ge_tensor) const {
  auto peer_node_2_out_anchor = std::make_pair(peer_node, out_data_anchor);
  if (peer_node->GetType() == ENTER || peer_node->GetType() == REFENTER) {
    const auto enter_in_data_anchor = peer_node->GetInDataAnchor(0);
    GE_CHECK_NOTNULL(enter_in_data_anchor);
    const auto enter_peer_out_data_anchor = enter_in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(enter_peer_out_data_anchor);
    peer_node = enter_peer_out_data_anchor->GetOwnerNode();
    peer_node_2_out_anchor.first = peer_node;
    peer_node_2_out_anchor.second = enter_peer_out_data_anchor;
  }
  const auto peer_op_desc = peer_node->GetOpDesc();
  GE_CHECK_NOTNULL(peer_op_desc);
  const auto peer_op_type = peer_op_desc->GetType();
  if (ConstantUtils::IsConstant(peer_op_desc)) {
    return ConstantUtils::GetWeight(peer_op_desc, static_cast<uint32_t>(peer_node_2_out_anchor.second->GetIdx()),
                                    ge_tensor) ? GRAPH_SUCCESS : GRAPH_FAILED;
  }
  if (peer_op_type == DATA) {
    auto parent_node_2_out_anchor = NodeUtils::GetParentInputAndAnchor(peer_node);
    while ((parent_node_2_out_anchor.first != nullptr) && (parent_node_2_out_anchor.first->GetType() == DATA)) {
      parent_node_2_out_anchor = NodeUtils::GetParentInputAndAnchor(parent_node_2_out_anchor.first);
    }
    if ((parent_node_2_out_anchor.first != nullptr) && (ConstantUtils::IsConstant(parent_node_2_out_anchor.first))) {
      const auto op_desc = parent_node_2_out_anchor.first->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      return ConstantUtils::GetWeight(op_desc, static_cast<uint32_t>(parent_node_2_out_anchor.second->GetIdx()),
                                      ge_tensor) ? GRAPH_SUCCESS : GRAPH_FAILED;
    }
  }
  return GRAPH_FAILED;
}

graphStatus OperatorImpl::GetInputConstData(const std::string &dst_name, Tensor &data) {
  GE_CHECK_NOTNULL(op_desc_);
  const auto index = op_desc_->GetInputIndexByName(dst_name);
  ConstGeTensorPtr ge_tensor = nullptr;
  if (GetInputConstData(static_cast<uint32_t>(index), ge_tensor) == GRAPH_SUCCESS) {
    data = TensorAdapter::GeTensor2Tensor(ge_tensor);
    return GRAPH_SUCCESS;
  }

  return GRAPH_FAILED;
}

graphStatus OperatorImpl::GetInputConstData(const uint32_t idx, ConstGeTensorPtr &ge_tensor) const {
  const auto node = GetNode();
  if (node == nullptr) {
    return GetInputConstDataOut(idx, ge_tensor);
  }

  // from runtime context
  const auto in_data_anchor = node->GetInDataAnchor(static_cast<int32_t>(idx));
  GE_CHECK_NOTNULL(in_data_anchor);
  const auto out_data_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(out_data_anchor);
  auto peer_node = out_data_anchor->GetOwnerNode();

  if (runtime_context_ != nullptr) {
    GeTensorPtr tensor_value = nullptr;
    if (runtime_context_->GetTensor(peer_node->GetOpDesc()->GetId(), out_data_anchor->GetIdx(),
                                    tensor_value) == GRAPH_SUCCESS) {
      ge_tensor = tensor_value;
      return GRAPH_SUCCESS;
    }
  }

  // For inner compute graph
  const auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  if (GetFromInputDesc(op_desc, static_cast<int32_t>(idx), ge_tensor) == GRAPH_SUCCESS) {
    return GRAPH_SUCCESS;
  }

  return GetFromPeerNode(peer_node, out_data_anchor, ge_tensor);
}

graphStatus OperatorImpl::GetInputConstDataOut(const uint32_t idx, ConstGeTensorPtr &ge_tensor) const {
  ge::OpIO out_handle("", 0, nullptr);
  if (GetInputImpl(idx, out_handle) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "index: %u get input impl failed", idx);
    GELOGE(FAILED, "[Get][InputImpl] failed, input index: %u", idx);
    return GRAPH_FAILED;
  }
  if (out_handle.GetOwner() != nullptr && out_handle.GetOwner()->GetOpDescImpl() != nullptr) {
    const auto &op_desc_impl_type = out_handle.GetOwner()->GetOpDescImpl()->GetType();
    if (op_desc_impl_type == CONSTANTOP || op_desc_impl_type == CONSTANT) {
      const auto op_desc = out_handle.GetOwner()->GetOpDescImpl();
      if (AttrUtils::GetTensor(op_desc, ATTR_NAME_WEIGHTS, ge_tensor)) {
        return GRAPH_SUCCESS;
      }
    }
  }
  return GRAPH_FAILED;
}

graphStatus OperatorImpl::GetInputConstDataOut(const std::string &dst_name, Tensor &data) const {
  ge::OpIO out_handle("", 0, nullptr);
  if (GetInputImpl(dst_name, out_handle) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "%s get input impl failed", dst_name.c_str());
    GELOGE(FAILED, "[Get][InputImpl] failed, dst_name:%s", dst_name.c_str());
    return GRAPH_FAILED;
  }
  if (out_handle.GetOwner() != nullptr && out_handle.GetOwner()->GetOpDescImpl() != nullptr) {
    const Operator const_op(out_handle.GetOwner());
    const auto &op_desc_impl_type = out_handle.GetOwner()->GetOpDescImpl()->GetType();
    if ((op_desc_impl_type == CONSTANTOP) || (op_desc_impl_type == CONSTANT)) {
      return const_op.GetAttr(ATTR_NAME_WEIGHTS, data);
    }
  }
  return GRAPH_FAILED;
}

bool OperatorImpl::InputIsSet(const std::string &name) {
  GE_CHK_BOOL_EXEC(op_desc_ != nullptr, REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr, check invalid.");
                   return false, "[Check][Param] op_desc_ is nullptr.");
  return op_desc_->InputIsSet(name);
}

std::string OperatorImpl::GetName() const {
  GE_CHK_BOOL_EXEC(op_desc_ != nullptr, REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr, check invalid.");
                   return std::string(), "[Check][Param] op_desc_ is nullptr.");
  return op_desc_->GetName();
}

GeTensorDesc OperatorImpl::GetInputDesc(const std::string &name) const {
  GE_CHK_BOOL_EXEC(op_desc_ != nullptr, REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr, check invalid.");
                   return GeTensorDesc(), "[Check][Param] op_desc_ is nullptr.");
  return op_desc_->GetInputDesc(name);
}

GeTensorDesc OperatorImpl::GetInputDesc(const uint32_t index) const {
  GE_CHK_BOOL_EXEC(op_desc_ != nullptr, REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr, check invalid.");
                   return GeTensorDesc(), "[Check][Param] op_desc_ is nullptr.");
  return op_desc_->GetInputDesc(index);
}

graphStatus OperatorImpl::UpdateInputDesc(const std::string &name, const GeTensorDesc &tensor_desc) {
  GE_CHK_BOOL_EXEC(op_desc_ != nullptr, REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr, check invalid.");
                   return GRAPH_FAILED, "[Check][Param] op_desc_ is nullptr.");

  return op_desc_->UpdateInputDesc(name, tensor_desc);
}

OutHandler OperatorImpl::GetOutput(const std::string &name) {
  GE_CHK_BOOL_EXEC(op_desc_ != nullptr, REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr, check invalid.");
                   return nullptr, "[Check][Param] op_desc_ is nullptr.");

  int32_t src_index = op_desc_->GetOutputIndexByName(name);
  GE_CHK_BOOL_EXEC(src_index >= 0,
                   REPORT_INNER_ERROR("E19999", "Find src index by name failed. name[%s]", name.c_str());
                   return nullptr, "[Get][OutputIndex] Find src index by name failed. name[%s]", name.c_str());
  const shared_ptr<OpIO> output_ptr = ComGraphMakeShared<OpIO>(name, src_index, shared_from_this());
  if (output_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "OpIO make shared failed");
    GELOGE(GRAPH_FAILED, "[Call][ComGraphMakeShared] OpIO make shared failed");
    return nullptr;
  }
  return output_ptr;
}

OutHandler OperatorImpl::GetOutput(uint32_t index) {
  GE_CHK_BOOL_EXEC(op_desc_ != nullptr, REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr, check invalid.");
                   return nullptr, "[Check][Param] op_desc_ is nullptr.");
  std::string name = op_desc_->GetOutputNameByIndex(index);
  if (name.empty()) {
    REPORT_INNER_ERROR("E19999", "Find src name by index failed. index[%u]", index);
    GELOGE(GRAPH_FAILED, "[Get][OutputName] Find src name by index failed. index[%u]", index);
    return nullptr;
  }
  const shared_ptr<OpIO> output_ptr = ComGraphMakeShared<OpIO>(name, index, shared_from_this());
  if (output_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "OpIO make shared failed");
    GELOGE(GRAPH_FAILED, "[Call][ComGraphMakeShared] OpIO make shared failed");
    return nullptr;
  }
  return output_ptr;
}

GeTensorDesc OperatorImpl::GetOutputDesc(const std::string &name) const {
  GE_CHK_BOOL_EXEC(op_desc_ != nullptr, REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr, check invalid.");
                   return GeTensorDesc(), "[Check][Param] op_desc_ is nullptr.");

  return op_desc_->GetOutputDesc(name);
}

GeTensorDesc OperatorImpl::GetOutputDesc(const uint32_t index) const {
  GE_CHK_BOOL_EXEC(op_desc_ != nullptr, REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr, check invalid.");
                   return GeTensorDesc(), "[Check][Param] op_desc_ is nullptr.");

  return op_desc_->GetOutputDesc(index);
}

graphStatus OperatorImpl::UpdateOutputDesc(const std::string &name, const GeTensorDesc &tensor_desc) {
  GE_CHK_BOOL_RET_STATUS(op_desc_ != nullptr, GRAPH_FAILED, "[Check][Param] op_desc is nullptr.");

  const auto res = op_desc_->UpdateOutputDesc(name, tensor_desc);
  if (res == GRAPH_SUCCESS) {
    for (const auto ol : output_links_[name]) {
      if (ol.GetOwner() == nullptr) {
        GELOGW("[Update][Check] %s get owner is nullptr", ol.GetName().c_str());
        continue;
      }
      GE_CHK_BOOL_RET_STATUS(ol.GetOwner()->UpdateInputDesc(ol.GetName(), tensor_desc) == GRAPH_SUCCESS, GRAPH_FAILED,
                             "[Update][InputDesc] Could not update next operator's input %s.", ol.GetName().c_str());
    }
  }
  return res;
}

size_t OperatorImpl::GetInputsSize() const {
  GE_IF_BOOL_EXEC(op_desc_ == nullptr, return 0);
  return op_desc_->GetInputsSize();
}

size_t OperatorImpl::GetOutputsSize() const {
  GE_IF_BOOL_EXEC(op_desc_ == nullptr, return 0U);
  return op_desc_->GetOutputsSize();
}

graphStatus OperatorImpl::SetAttr(const std::string &name, const AnyValue &&attr_value) {
  GE_CHK_BOOL_RET_STATUS(op_desc_ != nullptr, GRAPH_FAILED, "[Check][Param] op_desc is nullptr.");
  return op_desc_->SetAttr(name, std::move(attr_value));
}

graphStatus OperatorImpl::GetAttr(const std::string &name, AnyValue &attr_value) const {
  GE_CHK_BOOL_RET_STATUS(op_desc_ != nullptr, GRAPH_FAILED, "[Check][Param] op_desc is nullptr.");
  return op_desc_->GetAttr(name, attr_value);
}

OpDescPtr OperatorImpl::GetOpDescImpl() const {
  return op_desc_;
}

void OperatorImpl::UpdateLinkMapImpl(const std::string &src_name, const OpIO &op_dst) {
  const auto it_find = output_links_.find(src_name);
  if (it_find == output_links_.end()) {
    std::vector<OpIO> dsts{op_dst};
    (void)output_links_.insert(std::make_pair(src_name, dsts));
  } else {
    it_find->second.push_back(op_dst);
  }
}

Operator OperatorImpl::ToOperator() {
  return Operator(shared_from_this());
}

OpDescPtr OperatorImpl::GetOpDesc(const Operator &oprt) {
  GE_IF_BOOL_EXEC(oprt.operator_impl_ == nullptr, return nullptr);
  return oprt.operator_impl_->op_desc_;
}

void OperatorImpl::ClearOutputLinks() noexcept {
  output_links_.clear();
}

void OperatorImpl::ClearInputLinks() noexcept {
  input_link_.clear();
}

ge::ConstNodePtr OperatorImpl::GetNode() const {
  return node_;
}

void OperatorImpl::SetInferenceContext(const InferenceContextPtr &inference_context) {
  inference_context_ = inference_context;
}

InferenceContextPtr OperatorImpl::GetInferenceContext() const {
  return inference_context_;
}

void OperatorImpl::SubgraphRegister(const std::string &ir_name, const bool dynamic) {
  op_desc_->RegisterSubgraphIrName(ir_name, dynamic ? kDynamic : kStatic);
}

void OperatorImpl::SubgraphCountRegister(const std::string &ir_name, const uint32_t count) {
  if (op_desc_->GetSubgraphTypeByIrName(ir_name) == kStatic) {
    (void)op_desc_->AddSubgraphName(ir_name);
    subgraph_names_to_builders_[ir_name] = nullptr;
  } else {
    for (uint32_t i = 0U; i < count; ++i) {
      const std::string key_name = ir_name + std::to_string(i);
      (void)op_desc_->AddSubgraphName(key_name);
      subgraph_names_to_builders_[key_name] = nullptr;
    }
  }
}

void OperatorImpl::SetSubgraphBuilder(const std::string &ir_name, const uint32_t index,
                                      const SubgraphBuilder &builder) {
  std::string key_name = ir_name;
  if (op_desc_->GetSubgraphTypeByIrName(ir_name) == kDynamic) {
    key_name += std::to_string(index);
  }

  const auto it = subgraph_names_to_builders_.find(key_name);
  if (it == subgraph_names_to_builders_.end()) {
    REPORT_INNER_ERROR("E19999", "Failed to set subgraph builder for name %s index %u.", ir_name.c_str(), index);
    GELOGE(PARAM_INVALID, "[Check][Param] Failed to set subgraph builder for name %s index %u.", ir_name.c_str(),
           index);
    return;
  }
  it->second = builder;
}

SubgraphBuilder OperatorImpl::GetSubgraphBuilder(const std::string &ir_name, const uint32_t index) const {
  std::string key_name = ir_name;
  if (op_desc_->GetSubgraphTypeByIrName(ir_name) == kDynamic) {
    key_name += std::to_string(index);
  }

  return GetSubgraphBuilder(key_name);
}

SubgraphBuilder OperatorImpl::GetSubgraphBuilder(const std::string &name) const {
  const auto iter = subgraph_names_to_builders_.find(name);
  if (iter == subgraph_names_to_builders_.end()) {
    REPORT_INNER_ERROR("E19999", "Failed to get subgraph builder for name %s", name.c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Failed to get subgraph builder for name %s", name.c_str());
    return nullptr;
  }

  return iter->second;
}

std::vector<std::string> OperatorImpl::GetSubgraphNames() const {
  auto &ir_names = op_desc_->GetSubgraphIrNames();
  std::vector<std::string> names(ir_names.size());
  (void)std::transform(ir_names.begin(), ir_names.end(), names.begin(),
                       [](const std::pair<std::string, SubgraphType> &name_to_type) {
                         return name_to_type.first;
                       });
  return names;
}

size_t OperatorImpl::GetSubgraphNamesCount() const {
  return op_desc_->GetSubgraphIrNames().size();
}
}  // namespace ge
