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

#include "register/scope/scope_graph_impl.h"
#include <stack>
#include "external/register/register.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/string_util.h"
#include "graph/debug/ge_util.h"
#include "graph/ge_tensor.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/types.h"

namespace ge {
namespace {
using NodeEdges = std::map<int32_t, std::set<std::pair<std::string, int32_t>>>;
using GraphNodesInOut = std::unordered_map<std::string, std::pair<NodeEdges, NodeEdges>>;
const char_t *const kTfIdentityType = "Identity";
const char_t *const kTfConstType = "Const";
const char_t *const kNumerics = "0123456789";
const size_t kInputPartsSize = 2U;
const size_t kInputNodeName = 0U;
const size_t kPeerOutIndex = 1U;
const int32_t kControlSlot = -1;

Status DecomposeInputName(const std::string &input_name, std::string &node_name, int32_t &index, bool &is_control) {
  if (StringUtils::StartWith(input_name, "^")) {
    is_control = true;
    node_name = input_name.substr(1U);
    index = kControlSlot;
    return SUCCESS;
  }
  is_control = false;
  if (input_name.find(":") == std::string::npos) {
    node_name = input_name;
    index = 0;
    return SUCCESS;
  }
  const std::vector<std::string> parts = StringUtils::Split(input_name, ':');
  if (parts.size() != kInputPartsSize) {
    GELOGE(PARAM_INVALID, "Input name [%s] is invalid.", input_name.c_str());
    return PARAM_INVALID;
  }
  try {
    index = static_cast<int32_t>(std::stoi(parts[kPeerOutIndex]));
  } catch (std::invalid_argument &) {
    GELOGE(PARAM_INVALID, "Peer out index [%s] is invalid.", parts[kPeerOutIndex].c_str());
    return PARAM_INVALID;
  } catch (std::out_of_range &) {
    GELOGE(PARAM_INVALID, "Peer out index [%s] is out of range.", parts[kPeerOutIndex].c_str());
    return PARAM_INVALID;
  }
  if (index < 0) {
    GELOGE(PARAM_INVALID, "Peer out index [%d] is invalid.", index);
    return PARAM_INVALID;
  }
  node_name = parts[kInputNodeName];
  return SUCCESS;
}

void AddEdgeCtx(const std::string &peer_node_name, int32_t peer_index, int32_t curr_index, NodeEdges &node_edges) {
  (void)node_edges[curr_index].insert({peer_node_name, peer_index});
}

Status GetGraphDefInOutMap(domi::tensorflow::GraphDef *graph_def, GraphNodesInOut &in_out_map) {
  GE_CHECK_NOTNULL(graph_def);
  for (int32_t i = 0; i < graph_def->node_size(); i++) {
    const domi::tensorflow::NodeDef *node = graph_def->mutable_node(i);
    const std::string &node_name = node->name();
    int32_t input_index = 0;
    for (const auto &input : node->input()) {
      std::string peer_node_name;
      int32_t peer_out_index = 0;
      bool is_control = false;
      const auto ret = DecomposeInputName(input, peer_node_name, peer_out_index, is_control);
      if (ret != SUCCESS) {
        GELOGE(PARAM_INVALID, "Input[%s] of node[%s] is invalid.", input.c_str(), node_name.c_str());
        return PARAM_INVALID;
      }
      const int32_t curr_input_index = is_control ? kControlSlot : input_index++;
      AddEdgeCtx(peer_node_name, peer_out_index, curr_input_index, in_out_map[node_name].first);
      AddEdgeCtx(node_name, curr_input_index, peer_out_index, in_out_map[peer_node_name].second);
    }
  }
  return SUCCESS;
}

void GetInOutStr(const GraphNodesInOut &in_out_map, const string &node_name,
                 std::vector<std::string> &inputs, std::vector<std::string> &outputs) {
  const auto in_out_iter = in_out_map.find(node_name);
  if (in_out_iter == in_out_map.end()) {
    GELOGI("Not find input or output info for node:%s.", node_name.c_str());
    return;
  }
  const auto inputs_data = in_out_iter->second.first;
  for (const auto &input_data : inputs_data) {
    for (const auto &name_index : input_data.second) {
      std::string item = std::to_string(input_data.first) + ":" +  name_index.first +
                         ":" + std::to_string(name_index.second);
      inputs.push_back(item);
    }
  }

  const auto outputs_data = in_out_iter->second.second;
  for (const auto &output_data : outputs_data) {
    for (const auto &name_index : output_data.second) {
      std::string item = std::to_string(output_data.first) + ":" +  name_index.first +
        ":" + std::to_string(name_index.second);
      outputs.push_back(item);
    }
  }
}

Status SetNodeInputOutputAttr(const GraphNodesInOut &in_out_map, OperatorPtr &op) {
  GE_CHECK_NOTNULL(op);
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  GetInOutStr(in_out_map, op->GetName(), inputs, outputs);
  (void)op->SetAttr(ATTR_NAME_ORIGIN_GRAPH_NODE_INPUTS, inputs);
  (void)op->SetAttr(ATTR_NAME_ORIGIN_GRAPH_NODE_OUTPUTS, outputs);
  return SUCCESS;
}
}  // namespace

Status Scope::ScopeImpl::Init(const std::string &name, const std::string &sub_type, Scope *father_scope) {
  name_ = name;
  sub_type_ = sub_type;
  father_scope_ = father_scope;
  return SUCCESS;
}

Scope::ScopeImpl::~ScopeImpl() {
  for (auto &scope : sub_scopes_) {
    if (scope.second != nullptr) {
      delete scope.second;
      scope.second = nullptr;
    }
  }
}

void Scope::ScopeImpl::ClearTypeAndSubType() {
  sub_type_ = "";
  const std::vector<Scope *> &sub_scopes = GetAllSubScopes();
  for (auto &sub_scope : sub_scopes) {
    auto &impl = sub_scope->impl_;
    impl->SetSubType("");
  }
}

void Scope::ScopeImpl::AddNode(ge::OperatorPtr &node_def) {
  if (node_def == nullptr) {
    GELOGE(PARAM_INVALID, "Input node_def is nullptr.");
    return;
  }

  nodes_.push_back(node_def);
}

const std::unordered_map<std::string, ge::OperatorPtr> &Scope::ScopeImpl::AllNodesMap() {
  if (!all_nodes_map_.empty()) {
    return all_nodes_map_;
  }

  if (!nodes_.empty()) {
    for (const auto &node : nodes_) {
      all_nodes_map_.insert(std::pair<std::string, ge::OperatorPtr>(std::string(node->GetName()), node));
    }
  }
  const std::vector<Scope *> &scopes = GetAllSubScopes();
  for (auto &scope : scopes) {
    auto &impl = scope->impl_;
    const std::vector<ge::OperatorPtr> &sub_nodes = impl->Nodes();
    if (!sub_nodes.empty()) {
      for (auto sub_node : sub_nodes) {
        all_nodes_map_.insert(std::pair<std::string, ge::OperatorPtr>(std::string(sub_node->GetName()), sub_node));
      }
    }
  }
  return all_nodes_map_;
}

Scope *Scope::ScopeImpl::GetSubScope(const std::string &scope_name) const {
  const auto iter = sub_scopes_.find(scope_name);
  if (iter != sub_scopes_.end()) {
    return iter->second;
  }
  return nullptr;
}

const std::vector<Scope *> &Scope::ScopeImpl::GetAllSubScopes() {
  if (!all_sub_scopes_.empty()) {
    return all_sub_scopes_;
  }

  for (auto &iter : sub_scopes_) {
    Scope *scope = iter.second;
    all_sub_scopes_.push_back(scope);

    std::stack<Scope *> scopes;
    scopes.push(scope);
    while (!scopes.empty()) {
      Scope *scope = scopes.top();
      scopes.pop();
      auto &impl = scope->impl_;
      const std::unordered_map<std::string, Scope *> &sub_scopes = impl->GetSubScopes();
      for (auto &iter_sub : sub_scopes) {
        all_sub_scopes_.push_back(iter_sub.second);
        scopes.push(iter_sub.second);
      }
    }
  }
  return all_sub_scopes_;
}

int32_t Scope::ScopeImpl::GetOpTypeNum(const std::string &op_type) const {
  const auto iter = op_nums_.find(op_type);
  if (iter != op_nums_.end()) {
    return iter->second;
  } else {
    return -1;
  }
}

void Scope::ScopeImpl::OpsNumInc(const std::string &op_type) {
  const auto iter = op_nums_.find(op_type);
  if (iter != op_nums_.end()) {
    op_nums_[op_type] = iter->second + 1;
  } else {
    op_nums_[op_type] = 1;
  }
}

const std::string Scope::ScopeImpl::LastName() const {
  std::vector<std::string> names = ge::StringUtils::Split(name_, '/');
  // if vector size is less than 2, there is no multilevel directory, return origin name.
  if (names.size() < 2U) {
    GELOGI("Input name is already the last name, input name:%s.", name_.c_str());
    return name_;
  }
  const std::string last_name = names[names.size() - 2U];  // minus 2 to get the last name
  return ScopeImpl::TrimScopeIndex(last_name);
}

std::string Scope::ScopeImpl::TrimScopeIndex(const std::string &scope_name) {
  std::string scope_name_new = scope_name;
  // deal D_index, only keep name D
  const auto index = scope_name.find_last_of("_");
  if (index != std::string::npos) {
    // index_str after "_" is integer
    std::string index_str = scope_name.substr(index + 1U, scope_name.length());
    if (index_str.find_first_not_of(kNumerics) != std::string::npos) {
      return scope_name;
    }
    try {
      if (std::stoi(index_str.c_str()) > 0) {
        scope_name_new = scope_name.substr(0U, index);
      }
    } catch (std::invalid_argument &) {
      scope_name_new = scope_name;
    } catch (std::out_of_range &) {
      scope_name_new = scope_name;
    }
  }
  return scope_name_new;
}

Scope::Scope() {}

Status Scope::Init(const std::string &name, const std::string &sub_type, Scope *father_scope) {
  impl_ = std::unique_ptr<ScopeImpl>(new (std::nothrow) ScopeImpl);
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Make unique_ptr of ScopeImpl failed.");
    return ge::MEMALLOC_FAILED;
  }

  return impl_->Init(name, sub_type, father_scope);
}

Status Scope::Init(const char_t *name, const char_t *sub_type, Scope *father_scope) {
  std::string scope_name;
  std::string scope_sub_type;
  if (name != nullptr) {
    scope_name = name;
  }
  if (sub_type != nullptr) {
    scope_sub_type = sub_type;
  }
  impl_ = std::unique_ptr<ScopeImpl>(new (std::nothrow) ScopeImpl);
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Make unique_ptr of ScopeImpl failed.");
    return ge::MEMALLOC_FAILED;
  }

  return impl_->Init(scope_name, scope_sub_type, father_scope);
}

Scope::~Scope() {}

const std::string &Scope::Name() const {
  return impl_->Name();
}

Status Scope::Name(AscendString &name) const {
  name = AscendString(impl_->Name().c_str());
  return SUCCESS;
}

const std::string &Scope::SubType() const {
   return impl_->SubType();
}

Status Scope::SubType(AscendString &sub_type) const {
  sub_type = AscendString(impl_->SubType().c_str());
  return SUCCESS;
}

const std::unordered_map<std::string, ge::OperatorPtr> &Scope::AllNodesMap() const {
  return impl_->AllNodesMap();
}

Status Scope::AllNodesMap(std::unordered_map<AscendString, ge::OperatorPtr> &node_map) const {
  const std::unordered_map<std::string, ge::OperatorPtr> nodes = impl_->AllNodesMap();
  for (auto &node : nodes) {
    const AscendString tmp(node.first.c_str());
    node_map[tmp] = node.second;
  }
  return SUCCESS;
}

Scope *Scope::GetSubScope(const std::string &scope_name) const {
  return impl_->GetSubScope(scope_name);
}

Scope *Scope::GetSubScope(const char_t *scope_name) const {
  std::string str_scope_name;
  if (scope_name != nullptr) {
    str_scope_name = scope_name;
  }
  return impl_->GetSubScope(str_scope_name);
}

const std::string Scope::LastName() const {
  return impl_->LastName();
}

Status Scope::LastName(AscendString &name) const {
  name = AscendString(impl_->LastName().c_str());
  return SUCCESS;
}

const Scope *Scope::GetFatherScope() const {
  return impl_->GetFatherScope();
}

const std::vector<Scope *> &Scope::GetAllSubScopes() const {
  return impl_->GetAllSubScopes();
}

FusionScopesResult::InnerNodeInfo::InnerNodeInfoImpl::~InnerNodeInfoImpl() {
  operator_.BreakConnect();
}

std::string FusionScopesResult::InnerNodeInfo::InnerNodeInfoImpl::GetFullNodeName(const std::string &relative_name) {
  if (fusion_node_name_.empty()) {
    return relative_name;
  }
  return (fusion_node_name_.at(fusion_node_name_.size() - 1U) == '/') ? (fusion_node_name_ + relative_name)
                                                                        : (fusion_node_name_ + "/" + relative_name);
}

void FusionScopesResult::InnerNodeInfo::InnerNodeInfoImpl::InsertInput(const std::string &input_node,
                                                                       int32_t peer_out_idx) {
  std::string input_name = (input_node != kInputFromFusionScope) ? GetFullNodeName(input_node) : input_node;
  inner_node_inputs_.emplace_back(std::make_pair(input_name, peer_out_idx));
}

void FusionScopesResult::InnerNodeInfo::InnerNodeInfoImpl::InsertOutput(const std::string &output_node,
                                                                        int32_t peer_in_idx) {
  std::string output_name = (output_node != kOutputToFusionScope) ? GetFullNodeName(output_node) : output_node;
  inner_node_outputs_.emplace_back(std::make_pair(output_name, peer_in_idx));
}

ge::graphStatus FusionScopesResult::InnerNodeInfo::InnerNodeInfoImpl::BuildOperator() {
  operator_ = ge::OperatorFactory::CreateOperator(name_, type_);
  if (operator_.GetName() != name_) {
    GELOGE(ge::GRAPH_FAILED, "IR for op is not registered, op name:%s, op type:%s", name_.c_str(), type_.c_str());
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusionScopesResult::InnerNodeInfo::InnerNodeInfoImpl::SetInputFormat(const std::string &input_name,
                                                                                     const std::string &format) {
  ge::TensorDesc input_tesor_desc = operator_.GetInputDesc(input_name);
  const auto ge_format = ge::TypeUtils::SerialStringToFormat(format);
  input_tesor_desc.SetOriginFormat(ge_format);
  input_tesor_desc.SetFormat(ge_format);
  return operator_.UpdateInputDesc(input_name, input_tesor_desc);
}

ge::graphStatus FusionScopesResult::InnerNodeInfo::InnerNodeInfoImpl::SetOutputFormat(const std::string &output_name,
                                                                                      const std::string &format) {
  ge::TensorDesc output_tesor_desc = operator_.GetOutputDesc(output_name);
  const auto ge_format = ge::TypeUtils::SerialStringToFormat(format);
  output_tesor_desc.SetOriginFormat(ge_format);
  output_tesor_desc.SetFormat(ge_format);
  return operator_.UpdateOutputDesc(output_name, output_tesor_desc);
}

ge::graphStatus FusionScopesResult::InnerNodeInfo::InnerNodeInfoImpl::SetDynamicInputFormat(
    const std::string &input_name, uint32_t index, const std::string &format) {
  ge::TensorDesc input_tesor_desc = operator_.GetDynamicInputDesc(input_name, index);
  const auto ge_format = ge::TypeUtils::SerialStringToFormat(format);
  input_tesor_desc.SetOriginFormat(ge_format);
  input_tesor_desc.SetFormat(ge_format);
  return operator_.UpdateDynamicInputDesc(input_name, index, input_tesor_desc);
}

ge::graphStatus FusionScopesResult::InnerNodeInfo::InnerNodeInfoImpl::SetDynamicOutputFormat(
    const std::string &output_name, uint32_t index, const std::string &format) {
  ge::TensorDesc output_tesor_desc = operator_.GetDynamicOutputDesc(output_name, index);
  const auto ge_format = ge::TypeUtils::SerialStringToFormat(format);
  output_tesor_desc.SetOriginFormat(ge_format);
  output_tesor_desc.SetFormat(ge_format);
  return operator_.UpdateDynamicOutputDesc(output_name, index, output_tesor_desc);
}

FusionScopesResult::InnerNodeInfo::InnerNodeInfo(const std::string &fusion_node_name) {
  impl_ = std::unique_ptr<InnerNodeInfoImpl>(new (std::nothrow) InnerNodeInfoImpl(fusion_node_name));
}

FusionScopesResult::InnerNodeInfo::InnerNodeInfo(const char_t *fusion_node_name) {
  std::string str_fusion_node_name;
  if (fusion_node_name != nullptr) {
    str_fusion_node_name = fusion_node_name;
  }
  impl_ = std::unique_ptr<InnerNodeInfoImpl>(new (std::nothrow) InnerNodeInfoImpl(str_fusion_node_name));
}

FusionScopesResult::InnerNodeInfo::InnerNodeInfo(const std::string &fusion_node_name, const std::string &name,
                                                 const std::string &type) {
  impl_ = std::unique_ptr<InnerNodeInfoImpl>(new (std::nothrow) InnerNodeInfoImpl(fusion_node_name, name, type));
}

FusionScopesResult::InnerNodeInfo::InnerNodeInfo(const char_t *fusion_node_name, const char_t *name,
                                                 const char_t *type) {
  std::string node_name;
  if (fusion_node_name != nullptr) {
    node_name = fusion_node_name;
  }
  std::string str_name;
  if (name != nullptr) {
    str_name = name;
  }
  std::string str_type;
  if (type != nullptr) {
    str_type = type;
  }
  impl_ = std::unique_ptr<InnerNodeInfoImpl>(new (std::nothrow) InnerNodeInfoImpl(node_name,
                                                                                     str_name, str_type));
}

FusionScopesResult::InnerNodeInfo::InnerNodeInfo(FusionScopesResult::InnerNodeInfo &&other) noexcept
    : impl_(std::move(other.impl_)) {}

FusionScopesResult::InnerNodeInfo::~InnerNodeInfo() {}

FusionScopesResult::InnerNodeInfo &FusionScopesResult::InnerNodeInfo::operator=(
    FusionScopesResult::InnerNodeInfo &&other) noexcept {
  if (&other != this) {
    impl_ = std::move(other.impl_);
  }
  return *this;
}

FusionScopesResult::InnerNodeInfo &FusionScopesResult::InnerNodeInfo::SetName(const std::string &name) {
  if (impl_ != nullptr) {
    impl_->SetName(name);
  }
  return *this;
}

FusionScopesResult::InnerNodeInfo &FusionScopesResult::InnerNodeInfo::SetName(const char_t *name) {
  if ((impl_ != nullptr) && (name != nullptr)) {
    std::string str_name = name;
    impl_->SetName(str_name);
  }
  return *this;
}

FusionScopesResult::InnerNodeInfo &FusionScopesResult::InnerNodeInfo::SetType(const std::string &type) {
  if (impl_ != nullptr) {
    impl_->SetType(type);
  }
  return *this;
}

FusionScopesResult::InnerNodeInfo &FusionScopesResult::InnerNodeInfo::SetType(const char_t *type) {
  if (impl_ != nullptr && type != nullptr) {
    std::string str_type = type;
    impl_->SetType(str_type);
  }
  return *this;
}

FusionScopesResult::InnerNodeInfo &FusionScopesResult::InnerNodeInfo::InsertInput(const std::string &input_node,
                                                                                  int32_t peer_out_idx) {
  if (impl_ != nullptr) {
    impl_->InsertInput(input_node, peer_out_idx);
  }
  return *this;
}

FusionScopesResult::InnerNodeInfo &FusionScopesResult::InnerNodeInfo::InsertInput(const char_t *input_node,
                                                                                  int32_t peer_out_idx) {
  if ((impl_ != nullptr) && (input_node != nullptr)) {
    std::string str_input_node = input_node;
    impl_->InsertInput(str_input_node, peer_out_idx);
  }
  return *this;
}

FusionScopesResult::InnerNodeInfo &FusionScopesResult::InnerNodeInfo::InsertOutput(const std::string &output_node,
                                                                                   int32_t peer_in_idx) {
  if (impl_ != nullptr) {
    impl_->InsertOutput(output_node, peer_in_idx);
  }
  return *this;
}

FusionScopesResult::InnerNodeInfo &FusionScopesResult::InnerNodeInfo::InsertOutput(const char_t *output_node,
                                                                                   int32_t peer_in_idx) {
  if ((impl_ != nullptr) && (output_node != nullptr)) {
    std::string str_output_node = output_node;
    impl_->InsertOutput(str_output_node, peer_in_idx);
  }
  return *this;
}

ge::graphStatus FusionScopesResult::InnerNodeInfo::BuildInnerNode() {
  if (impl_ != nullptr) {
    return impl_->BuildOperator();
  }
  return ge::GRAPH_PARAM_INVALID;
}

ge::Operator *FusionScopesResult::InnerNodeInfo::MutableOperator() {
  if (impl_ != nullptr) {
    return impl_->MutableOperator();
  }
  return nullptr;
}

ge::graphStatus FusionScopesResult::InnerNodeInfo::SetInputFormat(const std::string &input_name,
                                                                  const std::string &format) {
  if (impl_ != nullptr) {
    return impl_->SetInputFormat(input_name, format);
  }
  return ge::GRAPH_PARAM_INVALID;
}

ge::graphStatus FusionScopesResult::InnerNodeInfo::SetInputFormat(const char_t *input_name,
                                                                  const char_t *format) {
  if ((impl_ != nullptr) && (input_name != nullptr) && (format != nullptr)) {
    const std::string str_input_name = input_name;
    const std::string str_format = format;
    return impl_->SetInputFormat(str_input_name, str_format);
  }
  return ge::GRAPH_PARAM_INVALID;
}

ge::graphStatus FusionScopesResult::InnerNodeInfo::SetOutputFormat(const std::string &output_name,
                                                                   const std::string &format) {
  if (impl_ != nullptr) {
    return impl_->SetOutputFormat(output_name, format);
  }
  return ge::GRAPH_PARAM_INVALID;
}

ge::graphStatus FusionScopesResult::InnerNodeInfo::SetOutputFormat(const char_t *output_name,
                                                                   const char_t *format) {
  if ((impl_ != nullptr) && (output_name != nullptr) && (format != nullptr)) {
    const std::string str_output_name = output_name;
    const std::string str_format = format;
    return impl_->SetOutputFormat(str_output_name, str_format);
  }
  return ge::GRAPH_PARAM_INVALID;
}

ge::graphStatus FusionScopesResult::InnerNodeInfo::SetDynamicInputFormat(const std::string &input_name, uint32_t index,
                                                                         const std::string &format) {
  if (impl_ != nullptr) {
    return impl_->SetDynamicInputFormat(input_name, index, format);
  }
  return ge::GRAPH_PARAM_INVALID;
}

ge::graphStatus FusionScopesResult::InnerNodeInfo::SetDynamicInputFormat(const char_t *input_name, uint32_t index,
                                                                         const char_t *format) {
  if ((impl_ != nullptr) && (input_name != nullptr) && (format != nullptr)) {
    const std::string str_input_name = input_name;
    const std::string str_format = format;
    return impl_->SetDynamicInputFormat(str_input_name, index, str_format);
  }
  return ge::GRAPH_PARAM_INVALID;
}

ge::graphStatus FusionScopesResult::InnerNodeInfo::SetDynamicOutputFormat(const std::string &output_name,
                                                                          uint32_t index, const std::string &format) {
  if (impl_ != nullptr) {
    return impl_->SetDynamicOutputFormat(output_name, index, format);
  }
  return ge::GRAPH_PARAM_INVALID;
}

ge::graphStatus FusionScopesResult::InnerNodeInfo::SetDynamicOutputFormat(const char_t *output_name,
                                                                          uint32_t index, const char_t *format) {
  if (impl_ != nullptr && output_name != nullptr && format != nullptr) {
    std::string str_output_name = output_name;
    std::string str_format = format;
    return impl_->SetDynamicOutputFormat(str_output_name, index, str_format);
  }
  return ge::GRAPH_PARAM_INVALID;
}

std::string FusionScopesResult::InnerNodeInfo::GetName() const {
  if (impl_ != nullptr) {
    return impl_->GetName();
  }
  return "";
}

ge::graphStatus FusionScopesResult::InnerNodeInfo::GetName(AscendString &name) const {
  if (impl_ != nullptr) {
    name = AscendString(impl_->GetName().c_str());
  }
  return GRAPH_SUCCESS;
}

std::string FusionScopesResult::InnerNodeInfo::GetType() const {
  if (impl_ != nullptr) {
    return impl_->GetType();
  }
  return "";
}

ge::graphStatus FusionScopesResult::InnerNodeInfo::GetType(AscendString &type) const {
  if (impl_ != nullptr) {
    type = AscendString(impl_->GetType().c_str());
  }
  return GRAPH_SUCCESS;
}

std::vector<std::pair<std::string, int32_t>> FusionScopesResult::InnerNodeInfo::GetInputs() const {
  const std::vector<std::pair<std::string, int32_t>> tmp;
  if (impl_ != nullptr) {
    return impl_->GetInputs();
  }
  return tmp;
}

ge::graphStatus FusionScopesResult::InnerNodeInfo::GetInputs(
    std::vector<std::pair<AscendString, int32_t>> &inputs) const {
  std::vector<std::pair<std::string, int32_t>> tmps;
  if (impl_ != nullptr) {
    tmps = impl_->GetInputs();
  }
  for (auto &tmp : tmps) {
    inputs.emplace_back(std::pair<AscendString, int32_t>(AscendString(tmp.first.c_str()), tmp.second));
  }
  return GRAPH_SUCCESS;
}

std::vector<std::pair<std::string, int32_t>> FusionScopesResult::InnerNodeInfo::GetOutputs() const {
  std::vector<std::pair<std::string, int32_t>> tmp;
  if (impl_ != nullptr) {
    return impl_->GetOutputs();
  }
  return tmp;
}

ge::graphStatus FusionScopesResult::InnerNodeInfo::GetOutputs(
    std::vector<std::pair<AscendString, int32_t>> &outputs) const {
  std::vector<std::pair<std::string, int32_t>> tmps;
  if (impl_ != nullptr) {
    tmps = impl_->GetOutputs();
  }
  for (auto &tmp : tmps) {
    outputs.emplace_back(std::pair<AscendString, int32_t>(tmp.first.c_str(), tmp.second));
  }
  return GRAPH_SUCCESS;
}

void FusionScopesResult::FusionScopesResultImpl::AddNodes(std::vector<ge::OperatorPtr> nodes) {
  (void)nodes_.insert(nodes_.end(), nodes.begin(), nodes.end());
}

void FusionScopesResult::FusionScopesResultImpl::InsertInputs(const std::string &inner_op_name,
                                                              const std::vector<int32_t> &index_map) {
  (void)inputs_.insert(make_pair(inner_op_name, index_map));
}
void FusionScopesResult::FusionScopesResultImpl::InsertOutputs(const std::string &inner_op_name,
                                                               const std::vector<int32_t> &index_map) {
  (void)outputs_.insert(make_pair(inner_op_name, index_map));
}

bool FusionScopesResult::FusionScopesResultImpl::FindNodes(const std::string &node_name) const {
  for (auto &node : nodes_) {
    if (node->GetName() == node_name) {
      return true;
    }
  }
  return false;
}

bool FusionScopesResult::FusionScopesResultImpl::FindScopes(const std::string &scope_name) const {
  for (auto &scope : scopes_) {
    if (scope->Name().length() < scope_name.length() && scope_name.find(scope->Name()) == 0) {
      return true;
    }
  }
  return false;
}

FusionScopesResult::InnerNodeInfo *FusionScopesResult::FusionScopesResultImpl::AddInnerNode(const std::string &name,
                                                                                            const std::string &type) {
  inner_node_infos_.emplace_back(InnerNodeInfo(name_, name, type));
  return &(inner_node_infos_[inner_node_infos_.size() - 1U]);
}

FusionScopesResult::InnerNodeInfo *FusionScopesResult::FusionScopesResultImpl::MutableRecentInnerNode() {
  size_t size = inner_node_infos_.size();
  if (size >= 1) {
    return &(inner_node_infos_[size - 1]);
  }
  return nullptr;
}

FusionScopesResult::InnerNodeInfo *FusionScopesResult::FusionScopesResultImpl::MutableInnerNode(uint32_t index) {
  if (static_cast<size_t>(index) < inner_node_infos_.size()) {
    return &(inner_node_infos_[static_cast<size_t>(index)]);
  }
  return nullptr;
}

FusionInnerNodesInfo FusionScopesResult::FusionScopesResultImpl::GetInnerNodesInfo() {
  FusionInnerNodesInfo nodes_info;
  for (auto &info : inner_node_infos_) {
    nodes_info.emplace_back(
        std::make_tuple(info.GetName(), info.GetType(), info.GetInputs(), info.GetOutputs(), info.MutableOperator()));
  }
  return nodes_info;
}

ge::graphStatus FusionScopesResult::FusionScopesResultImpl::CheckInnerNodesInfo() {
  size_t input_from_scope = 0U;
  size_t output_to_scope = 0U;
  std::set<std::string> name_set;
  for (const auto &info : inner_node_infos_) {
    if (!(name_set.insert(info.GetName()).second)) {
      GELOGE(ge::GRAPH_PARAM_INVALID, "There are duplicate internal node name, please check.");
      return ge::GRAPH_PARAM_INVALID;
    }
    for (const auto &input : info.GetInputs()) {
      input_from_scope += (input.first == kInputFromFusionScope) ? 1 : 0;
    }
    for (const auto &input : info.GetOutputs()) {
      output_to_scope += (input.first == kOutputToFusionScope) ? 1 : 0;
    }
  }
  size_t scope_input = 0U;
  size_t scope_output = 0U;
  for (const auto &input : inputs_) {
    for (const auto &idx : input.second) {
      scope_input += (idx != kFusionDisableIndex) ? 1 : 0;
    }
  }
  for (const auto &output : outputs_) {
    for (const auto &idx : output.second) {
      scope_output += (idx != kFusionDisableIndex) ? 1 : 0;
    }
  }
  if ((input_from_scope != scope_input) || (output_to_scope != scope_output)) {
    GELOGE(ge::GRAPH_PARAM_INVALID,
           "Input or Output mismatched, please check. "
           "Inner input_from_scope:%zu, scope input:%zu, "
           "inner output_to_scope:%zu, scope output:%zu.",
           input_from_scope, scope_input, output_to_scope, scope_output);
    return ge::GRAPH_PARAM_INVALID;
  }
  return ge::GRAPH_SUCCESS;
}

FusionScopesResult::FusionScopesResult() {}

Status FusionScopesResult::Init() {
  impl_ = std::unique_ptr<FusionScopesResultImpl>(new (std::nothrow) FusionScopesResultImpl);
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Make unique_ptr of FusionScopesResultImpl failed.");
    return ge::MEMALLOC_FAILED;
  }

  return SUCCESS;
}

FusionScopesResult::~FusionScopesResult() {}

void FusionScopesResult::SetName(const std::string &name) {
  impl_->SetName(name);
}

void FusionScopesResult::SetName(const char_t *name) {
  std::string str_name;
  if (name != nullptr) {
    str_name = name;
  }
  impl_->SetName(str_name);
}

void FusionScopesResult::SetType(const std::string &type) {
  impl_->SetType(type);
}

void FusionScopesResult::SetType(const char_t *type) {
  std::string str_type;
  if (type != nullptr) {
    str_type = type;
  }
  impl_->SetType(str_type);
}

void FusionScopesResult::SetDescription(const std::string &description) {
  impl_->SetDescription(description);
}

void FusionScopesResult::SetDescription(const char_t *description) {
  std::string str_desc;
  if (description != nullptr) {
    str_desc = description;
  }
  impl_->SetDescription(str_desc);
}

const std::string &FusionScopesResult::Name() const {
  return impl_->Name();
}

const Status FusionScopesResult::Name(AscendString &name) const {
  name = AscendString(impl_->Name().c_str());
  return SUCCESS;
}

const std::vector<ge::OperatorPtr> &FusionScopesResult::Nodes() const {
  return impl_->Nodes();
}

void FusionScopesResult::InsertInputs(const std::string &inner_op_name, const std::vector<int32_t> &index_map) {
  impl_->InsertInputs(inner_op_name, index_map);
}

void FusionScopesResult::InsertInputs(const char_t *inner_op_name, const std::vector<int32_t> &index_map) {
  std::string op_name;
  if (inner_op_name != nullptr) {
    op_name = inner_op_name;
  }
  impl_->InsertInputs(op_name, index_map);
}

void FusionScopesResult::InsertOutputs(const std::string &inner_op_name, const std::vector<int32_t> &index_map) {
  impl_->InsertOutputs(inner_op_name, index_map);
}

void FusionScopesResult::InsertOutputs(const char_t *inner_op_name, const std::vector<int32_t> &index_map) {
  std::string op_name;
  if (inner_op_name != nullptr) {
    op_name = inner_op_name;
  }
  impl_->InsertOutputs(op_name, index_map);
}

FusionScopesResult::InnerNodeInfo *FusionScopesResult::AddInnerNode(const std::string &name, const std::string &type) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "FusionScopesResult is not properly initialized.");
    return nullptr;
  }
  return impl_->AddInnerNode(name, type);
}

FusionScopesResult::InnerNodeInfo *FusionScopesResult::AddInnerNode(const char_t *name, const char_t *type) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "FusionScopesResult is not properly initialized.");
    return nullptr;
  }
  std::string str_name;
  if (name != nullptr) {
    str_name = name;
  }
  std::string str_type;
  if (type != nullptr) {
    str_type = type;
  }
  return impl_->AddInnerNode(str_name, str_type);
}

FusionScopesResult::InnerNodeInfo *FusionScopesResult::MutableRecentInnerNode() {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "FusionScopesResult is not properly initialized.");
    return nullptr;
  }
  return impl_->MutableRecentInnerNode();
}

FusionScopesResult::InnerNodeInfo *FusionScopesResult::MutableInnerNode(uint32_t index) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "FusionScopesResult is not properly initialized.");
    return nullptr;
  }
  return impl_->MutableInnerNode(index);
}

ge::graphStatus FusionScopesResult::CheckInnerNodesInfo() {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "FusionScopesResult is not properly initialized.");
    return ge::GRAPH_PARAM_INVALID;
  }
  return impl_->CheckInnerNodesInfo();
}

Status ScopeTree::ScopeTreeImpl::Init() {
  root_ = new (std::nothrow) Scope();
  if (root_ == nullptr) {
    GELOGE(FAILED, "Alloc root scope failed.");
    return FAILED;
  }
  if (root_->Init("root") != SUCCESS) {
    GELOGE(FAILED, "Init root scope failed.");
    return FAILED;
  }
  scopes_.push_back(root_);
  return SUCCESS;
}

ScopeTree::ScopeTreeImpl::~ScopeTreeImpl() {
  if (root_ != nullptr) {
    delete root_;
    root_ = nullptr;
  }
}

void ScopeTree::ScopeTreeImpl::AddNodeToScope(ge::OperatorPtr &node_def) {
  if (node_def == nullptr) {
    GELOGE(PARAM_INVALID, "Input node_def is nullptr.");
    return;
  }
  const std::string &node_name = node_def->GetName();
  Scope *super_scope = root_;

  const std::vector<std::string> scopes = SplitNodeName(node_name, '/');
  for (uint32_t i = 0U; i < scopes.size(); ++i) {
    auto &impl = super_scope->impl_;
    impl->OpsNumInc(node_def->GetOpType());

    if (i == (scopes.size() - 1)) {
      impl->AddNode(node_def);
    } else {
      Scope *sub_scope = impl->GetSubScope(scopes[i]);
      if (sub_scope == nullptr) {
        sub_scope = new (std::nothrow) Scope();
        if (sub_scope == nullptr) {
          GELOGE(FAILED, "Alloc Scope failed.");
          return;
        }
        if (sub_scope->Init(scopes[i], "", super_scope) != SUCCESS) {
          GELOGE(FAILED, "Init Scope failed.");
          delete sub_scope;
          sub_scope = nullptr;
          return;
        }
        scopes_.push_back(sub_scope);
        impl->AddSubScope(sub_scope);
      }
      super_scope = sub_scope;
    }
  }
}

std::vector<std::string> ScopeTree::ScopeTreeImpl::SplitNodeName(const std::string &node_name,
                                                                 const char_t delim) const {
  std::vector<std::string> items;
  std::vector<std::string> scopes;
  if (node_name == "") {
    return items;
  }

  items = ge::StringUtils::Split(node_name, delim);
  std::string scope;
  for (uint32_t i = 0U; i < items.size(); ++i) {
    if (items[i].length() == 0U) {
      continue;
    }

    if (i == 0U) {
      scope = items[i];
    } else {
      scope = scope + items[i];
    }

    if (i != (items.size() - 1U)) {
      scope = scope + delim;
    }

    scopes.push_back(scope);
  }

  return scopes;
}

ScopeTree::ScopeTree() {}

ScopeTree::~ScopeTree() {}

Status ScopeTree::Init() {
  impl_ = std::unique_ptr<ScopeTreeImpl>(new (std::nothrow) ScopeTreeImpl);
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Make unique_ptr of FusionScopesResultImpl failed.");
    return ge::MEMALLOC_FAILED;
  }
  return impl_->Init();
}

const std::vector<Scope *> &ScopeTree::GetAllScopes() const {
  return impl_->GetAllScopes();
}

Status ScopeGraph::ScopeGraphImpl::Init() {
  scope_tree_ = new (std::nothrow) ScopeTree();
  if (scope_tree_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Alloc scope tree failed.");
    return ge::MEMALLOC_FAILED;
  }
  const Status ret = scope_tree_->Init();
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Scope tree init failed.");
    return FAILED;
  }
  return SUCCESS;
}

ScopeGraph::ScopeGraphImpl::~ScopeGraphImpl() {
  if (scope_tree_ != nullptr) {
    delete scope_tree_;
    scope_tree_ = nullptr;
  }

  for (auto &fusion_result : fusion_results_) {
    if (fusion_result.second != nullptr) {
      delete fusion_result.second;
      fusion_result.second = nullptr;
    }
  }

  for (const auto &item : nodes_map_) {
    item.second->BreakConnect();
  }
}

void ScopeGraph::ScopeGraphImpl::BuildScopeGraph(domi::tensorflow::GraphDef *graph_def) {
  if (graph_def == nullptr) {
    GELOGE(PARAM_INVALID, "Input graph_def is nullptr.");
    return;
  }
  GraphNodesInOut graph_nodes_in_out;
  const auto status = GetGraphDefInOutMap(graph_def, graph_nodes_in_out);
  if (status != SUCCESS) {
    GELOGE(FAILED, "Failed to get node input output map for graph.");
    return;
  }

  for (int32_t i = 0; i < graph_def->node_size(); ++i) {
    const domi::tensorflow::NodeDef *const node_def = graph_def->mutable_node(i);
    ge::OperatorPtr op(new (std::nothrow) ge::Operator(node_def->name(), node_def->op()));
    if (op == nullptr) {
      GELOGE(ge::MEMALLOC_FAILED, "Make shared_ptr<Operator> falied.");
      return;
    }
    const auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(*op);
    Status ret = domi::AutoMappingFn(node_def, *op);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Op: %s call auto mapping function failed.", op_desc->GetName().c_str());
      return;
    }

    for (int32_t j = 0; j < node_def->input_size(); j++) {
      ge::GeTensorDesc tensor_desc;
      tensor_desc.SetName(node_def->input(j));
      (void)op_desc->AddInputDesc(tensor_desc);
    }
    ret = SetNodeInputOutputAttr(graph_nodes_in_out, op);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Failed to set input output attr, op:%s.", op->GetName().c_str());
      return;
    }

    nodes_map_.emplace(op->GetName(), op);
    if (op->GetOpType() != kTfIdentityType || op->GetOpType() != kTfConstType) {
      auto &impl = scope_tree_->impl_;
      impl->AddNodeToScope(op);
    }
  }
}

void ScopeGraph::ScopeGraphImpl::AddFusionScopesResult(FusionScopesResult *result) {
  if (result == nullptr) {
    GELOGE(PARAM_INVALID, "Input params invalid, result is nullptr.");
    return;
  }
  fusion_results_[result->Name()] = result;
}

bool ScopeGraph::ScopeGraphImpl::IsFusionOpChild(const std::string &node_name,
                                                 std::vector<ScopeFusionOpInfo> &info_list) {
  bool find = false;
  for (auto &fusion_result : fusion_results_) {
    const FusionScopesResult *const fusion_node = fusion_result.second;
    auto &impl = fusion_node->impl_;

    if (impl->FindNodes(node_name) || impl->FindScopes(node_name)) {
      ScopeFusionOpInfo info;
      info.fusion_node_name = fusion_node->Name();
      info.fusion_op_type = impl->Type();
      info.node_name = node_name;
      info.description = impl->Description();
      info.scope_pass = true;
      info_list.push_back(info);

      find = true;
    }
  }

  return find;
}

bool ScopeGraph::ScopeGraphImpl::FusionOpChildIgnore(const ScopeFusionOpInfo &info) {
  if ((!(GetFusionResultInputOrOutput(info, true).empty())) ||
      (!(GetFusionResultInputOrOutput(info, false).empty()))) {
    return false;
  }
  return true;
}

std::vector<int32_t> ScopeGraph::ScopeGraphImpl::GetFusionResultInputOrOutput(const ScopeFusionOpInfo &info,
                                                                              bool input) {
  std::vector<int32_t> indexs;
  const auto fusion_iter = fusion_results_.find(info.fusion_node_name);
  if (fusion_iter == fusion_results_.end()) {
    GELOGE(FAILED, "Get fusion result failed, not found node:%s", info.fusion_node_name.c_str());
    return indexs;
  }

  const FusionScopesResult *const fusion_node = fusion_iter->second;
  std::map<std::string, std::vector<int32_t>> inout_map;
  auto &impl = fusion_node->impl_;
  if (input) {
    inout_map = impl->GetInputs();
  } else {
    inout_map = impl->GetOutputs();
  }

  for (auto &iter : inout_map) {
    const std::string input_name = iter.first;
    const std::string op_name = (info.node_name.length() > input_name.length())
                         ? info.node_name.substr(info.node_name.length() - input_name.length())
                         : info.node_name;
    if (input_name == op_name) {
      (void)indexs.insert(indexs.end(), iter.second.begin(), iter.second.end());
      break;
    }
  }

  return indexs;
}

bool ScopeGraph::ScopeGraphImpl::IsFusionOp(const domi::tensorflow::NodeDef *node_def) {
  if (node_def == nullptr) {
    GELOGE(PARAM_INVALID, "Input node_def is nullptr.");
    return false;
  }
  for (auto &fusion_result : fusion_results_) {
    const FusionScopesResult *const fusion_node = fusion_result.second;
    auto &impl = fusion_node->impl_;
    if (impl->Type() == node_def->op() && fusion_node->Name() == node_def->name()) {
      return true;
    }
  }
  return false;
}

Status ScopeGraph::ScopeGraphImpl::GetInputOrOutputIndex(const ScopeFusionOpInfo &info, int32_t old_index,
                                                         bool input, int32_t &new_index) {
  if (old_index == -1) {
    new_index = -1;
    return SUCCESS;
  }

  std::vector<int32_t> indexs = GetFusionResultInputOrOutput(info, input);
  GELOGD("GetNodeindex, node_name:%s, fusion_node_name:%s, fusion_op_type:%s, old_index:%d, size:%zu.",
         info.node_name.c_str(), info.fusion_node_name.c_str(), info.fusion_op_type.c_str(), old_index, indexs.size());
  if (static_cast<int32_t>(indexs.size()) < (old_index + 1)) {
    GELOGD("GetNodeindex fusionDisableIndex, node_name:%s, fusion_node_name:%s, fusion_op_type:%s, old_index:%d .",
           info.node_name.c_str(), info.fusion_node_name.c_str(), info.fusion_op_type.c_str(), old_index);
    new_index = kFusionDisableIndex;
  } else {
    new_index = indexs[static_cast<int32_t>(old_index)];
  }
  GELOGD("RESULT: new index:%d.", new_index);
  return SUCCESS;
}

FusionScopesResult *ScopeGraph::ScopeGraphImpl::GetFusionScopesResults(
    const domi::tensorflow::NodeDef *node_def) const {
  if (node_def == nullptr) {
    return nullptr;
  }
  return GetFusionScopesResults(node_def->name());
}

FusionScopesResult *ScopeGraph::ScopeGraphImpl::GetFusionScopesResults(const string &node_name) const {
  const auto iter = fusion_results_.find(node_name);
  if (iter != fusion_results_.end()) {
    return iter->second;
  } else {
    return nullptr;
  }
}

ScopeGraph::ScopeGraph() {}

ScopeGraph::~ScopeGraph() {}

Status ScopeGraph::Init() {
  impl_ = std::unique_ptr<ScopeGraphImpl>(new (std::nothrow) ScopeGraphImpl);
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Make unique_ptr of ScopeGraphImpl failed.");
    return ge::MEMALLOC_FAILED;
  }
  return impl_->Init();
}

const ScopeTree *ScopeGraph::GetScopeTree() const {
  return impl_->GetScopeTree();
}

const std::unordered_map<std::string, ge::OperatorPtr> &ScopeGraph::GetNodesMap() const {
  return impl_->GetNodesMap();
}

Status ScopeGraph::GetNodesMap(std::unordered_map<AscendString, ge::OperatorPtr> &nodes_map) const {
  std::unordered_map<std::string, ge::OperatorPtr> tmps;
  if (impl_ != nullptr) {
    tmps = impl_->GetNodesMap();
  }
  for (auto &tmp : tmps) {
    AscendString node(tmp.first.c_str());
    nodes_map[node] = tmp.second;
  }
  return SUCCESS;
}
}  // namespace ge
