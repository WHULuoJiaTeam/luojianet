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

#include "register/scope/scope_pattern_impl.h"
#include "register/scope/scope_graph_impl.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_util.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/types.h"

namespace ge {
ScopeAttrValue::ScopeAttrValue() {
  impl_ = std::unique_ptr<ScopeAttrValueImpl>(new (std::nothrow) ScopeAttrValueImpl);
}

ScopeAttrValue::ScopeAttrValue(ScopeAttrValue const &attr_value) {
  impl_ = std::unique_ptr<ScopeAttrValueImpl>(new (std::nothrow) ScopeAttrValueImpl);
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "ScopeAttrValue is not properly initialized.");
    return;
  }
  impl_->SetIntValue(attr_value.impl_->GetIntValue());
  impl_->SetFloatValue(attr_value.impl_->GetFloatValue());
  impl_->SetStringValue(attr_value.impl_->GetStrValue());
  impl_->SetBoolValue(attr_value.impl_->GetBoolValue());
}

ScopeAttrValue &ScopeAttrValue::operator=(ScopeAttrValue const &attr_value) {
  if (&attr_value == this) {
    return *this;
  }
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "ScopeAttrValue is not properly initialized.");
    return *this;
  }
  impl_->SetIntValue(attr_value.impl_->GetIntValue());
  impl_->SetFloatValue(attr_value.impl_->GetFloatValue());
  impl_->SetStringValue(attr_value.impl_->GetStrValue());
  impl_->SetBoolValue(attr_value.impl_->GetBoolValue());
  return *this;
}

ScopeAttrValue::~ScopeAttrValue() {}

void ScopeAttrValue::SetIntValue(int64_t value) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Failed to invoke SetIntValue(), ScopeAttrValue is not properly initialized.");
    return;
  }
  impl_->SetIntValue(value);
}

void ScopeAttrValue::SetFloatValue(float32_t value) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Failed to invoke SetFloatValue(), ScopeAttrValue is not properly initialized.");
    return;
  }
  impl_->SetFloatValue(value);
}

void ScopeAttrValue::SetStringValue(std::string value) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Failed to invoke SetStringValue(), ScopeAttrValue is not properly initialized.");
    return;
  }
  impl_->SetStringValue(value);
}

void ScopeAttrValue::SetStringValue(const char_t *value) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Failed to invoke SetStringValue(), ScopeAttrValue is not properly initialized.");
    return;
  }
  std::string str_value;
  if (value != nullptr) {
    str_value = value;
  }
  impl_->SetStringValue(str_value);
}

void ScopeAttrValue::SetBoolValue(bool value) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Failed to invoke SetBoolValue(), ScopeAttrValue is not properly initialized.");
    return;
  }
  impl_->SetBoolValue(value);
}

bool NodeOpTypeFeature::NodeOpTypeFeatureImpl::Match(const Scope *scope) {
  if (scope == nullptr) {
    GELOGE(PARAM_INVALID, "Input scope is nullptr.");
    return false;
  }
  auto &impl = scope->impl_;

  if (step_ == 0) {
    if (impl->GetOpTypeNum(node_type_) == num_) {
      GELOGI("NodeOpTypeFeature, node type:%s, num:%d, match scope:%s",
             node_type_.c_str(), num_, scope->Name().c_str());
      return true;
    }
  } else {
    if ((impl->GetOpTypeNum(node_type_) != -1) && ((impl->GetOpTypeNum(node_type_) % step_) == num_)) {
      GELOGI("NodeOpTypeFeature, node type:%s, num:%d, match scope:%s",
             node_type_.c_str(), num_, scope->Name().c_str());
      return true;
    }
  }

  return false;
}

NodeOpTypeFeature::NodeOpTypeFeature(std::string nodeType, int32_t num, int32_t step)
    : ScopeBaseFeature() {
  impl_ = std::unique_ptr<NodeOpTypeFeatureImpl>(new (std::nothrow) NodeOpTypeFeatureImpl(nodeType, num, step));
}

NodeOpTypeFeature::NodeOpTypeFeature(const char_t *node_type, int32_t num, int32_t step)
    : ScopeBaseFeature() {
  std::string op_type;
  if (node_type != nullptr) {
    op_type = node_type;
  }
  impl_ = std::unique_ptr<NodeOpTypeFeatureImpl>(new (std::nothrow) NodeOpTypeFeatureImpl(op_type, num, step));
}

NodeOpTypeFeature::NodeOpTypeFeature(NodeOpTypeFeature const &feature) : ScopeBaseFeature() {
  impl_ = std::unique_ptr<NodeOpTypeFeatureImpl>(new (std::nothrow) NodeOpTypeFeatureImpl(feature.impl_->node_type_,
                                                                                          feature.impl_->num_,
                                                                                          feature.impl_->step_));
}

NodeOpTypeFeature &NodeOpTypeFeature::operator=(NodeOpTypeFeature const &feature) {
  if (&feature == this) {
    return *this;
  }

  impl_->node_type_ = feature.impl_->node_type_;
  impl_->num_ = feature.impl_->num_;
  impl_->step_ = feature.impl_->step_;
  return *this;
}

NodeOpTypeFeature::~NodeOpTypeFeature() {}

bool NodeOpTypeFeature::Match(const Scope *scope) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Failed to invoke Match(), NodeOpTypeFeature is not properly initialized.");
    return false;
  }

  return impl_->Match(scope);
}

bool NodeAttrFeature::NodeAttrFeatureImpl::Match(const Scope *scope) {
  if (scope == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Input scope is nullptr.");
    return false;
  }
  auto &impl = scope->impl_;
  const std::vector<ge::OperatorPtr> &nodes = impl->Nodes();
  for (auto &node_op : nodes) {
    if (node_type_ != node_op->GetOpType()) {
      continue;
    }
    const auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(*node_op);
    if (op_desc == nullptr) {
      GELOGE(ge::PARAM_INVALID, "Op desc is nullptr.");
      return false;
    }

    Status result = SUCCESS;
    switch (datatype_) {
      case ge::DT_FLOAT:
        result = CheckNodeAttrFeatureData(0.0F, op_desc, scope);
        break;
      case ge::DT_INT32:
        result = CheckNodeAttrFeatureData(static_cast<int64_t>(0), op_desc, scope);
        break;
      case ge::DT_STRING:
        result = CheckNodeAttrFeatureData("", op_desc, scope);
        break;
      case ge::DT_BOOL:
        result = CheckNodeAttrFeatureData(false, op_desc, scope);
        break;
      default:
        break;
    }
    if (result != FAILED) {
      return (result == PARAM_INVALID) ? false : true;
    }
  }
  return false;
}

Status NodeAttrFeature::NodeAttrFeatureImpl::CheckNodeAttrFeatureData(const bool init_value,
                                                                      const ge::OpDescPtr &op_desc,
                                                                      const Scope *const scope) {
  bool value = init_value;
  if (!ge::AttrUtils::GetBool(op_desc, attr_name_, value)) {
    GELOGE(ge::PARAM_INVALID, "op:%s %s attr is null", op_desc->GetName().c_str(), attr_name_.c_str());
    return PARAM_INVALID;
  }
  if (attr_value_.impl_->GetBoolValue() == value) {
    GELOGI("NodeAttrFeature, match scope:%s", scope->Name().c_str());
    return SUCCESS;
  }
  return FAILED;
}

Status NodeAttrFeature::NodeAttrFeatureImpl::CheckNodeAttrFeatureData(const std::string init_value,
                                                                      const ge::OpDescPtr &op_desc,
                                                                      const Scope *const scope) {
  std::string value = init_value;
  if (!ge::AttrUtils::GetStr(op_desc, attr_name_, value)) {
    GELOGE(ge::PARAM_INVALID, "op:%s %s attr is null", op_desc->GetName().c_str(), attr_name_.c_str());
    return PARAM_INVALID;
  }
  if (attr_value_.impl_->GetStrValue() == value) {
    GELOGI("NodeAttrFeature, match scope:%s", scope->Name().c_str());
    return SUCCESS;
  }
  return FAILED;
}

Status NodeAttrFeature::NodeAttrFeatureImpl::CheckNodeAttrFeatureData(const int64_t init_value,
                                                                      const ge::OpDescPtr &op_desc,
                                                                      const Scope *const scope) {
  int64_t value = init_value;
  if (!ge::AttrUtils::GetInt(op_desc, attr_name_, value)) {
    GELOGE(ge::PARAM_INVALID, "op:%s %s attr is null", op_desc->GetName().c_str(), attr_name_.c_str());
    return PARAM_INVALID;
  }
  if (attr_value_.impl_->GetIntValue() == value) {
    GELOGI("NodeAttrFeature, match scope:%s", scope->Name().c_str());
    return SUCCESS;
  }
  return FAILED;
}

Status NodeAttrFeature::NodeAttrFeatureImpl::CheckNodeAttrFeatureData(const float32_t init_value,
                                                                      const ge::OpDescPtr &op_desc,
                                                                      const Scope *const scope) {
  float32_t value = init_value;
  if (!ge::AttrUtils::GetFloat(op_desc, attr_name_, value)) {
    GELOGE(ge::PARAM_INVALID, "op:%s %s attr is null", op_desc->GetName().c_str(), attr_name_.c_str());
    return PARAM_INVALID;
  }

  if (FloatIsEqual(attr_value_.impl_->GetFloatValue(), value)) {
    GELOGI("NodeAttrFeature, match scope:%s", scope->Name().c_str());
    return SUCCESS;
  }
  return FAILED;
}

NodeAttrFeature::NodeAttrFeature(std::string nodeType, std::string attr_name,
                                 ge::DataType datatype, ScopeAttrValue &attr_value)
    : ScopeBaseFeature() {
  impl_ = std::unique_ptr<NodeAttrFeatureImpl>(new (std::nothrow) NodeAttrFeatureImpl(nodeType, attr_name,
                                                                                      datatype, attr_value));
}

NodeAttrFeature::NodeAttrFeature(const char_t *node_type, const char_t *attr_name,
                                 ge::DataType data_type, ScopeAttrValue &attr_value)
    : ScopeBaseFeature() {
  std::string str_node_type;
  if (node_type != nullptr) {
    str_node_type = node_type;
  }
  std::string str_attr_name;
  if (attr_name != nullptr) {
    str_attr_name = attr_name;
  }
  impl_ = std::unique_ptr<NodeAttrFeatureImpl>(new (std::nothrow) NodeAttrFeatureImpl(str_node_type, str_attr_name,
                                                                                      data_type, attr_value));
}

NodeAttrFeature::NodeAttrFeature(NodeAttrFeature const &feature) : ScopeBaseFeature() {
  impl_ = std::unique_ptr<NodeAttrFeatureImpl>(new (std::nothrow) NodeAttrFeatureImpl(feature.impl_->node_type_,
                                                                                      feature.impl_->attr_name_,
                                                                                      feature.impl_->datatype_,
                                                                                      feature.impl_->attr_value_));
}

NodeAttrFeature &NodeAttrFeature::operator=(NodeAttrFeature const &feature) {
  if (&feature == this) {
    return *this;
  }
  impl_->node_type_ = feature.impl_->node_type_;
  impl_->attr_name_ = feature.impl_->attr_name_;
  impl_->datatype_ = feature.impl_->datatype_;
  impl_->attr_value_ = feature.impl_->attr_value_;
  return *this;
}

NodeAttrFeature::~NodeAttrFeature() {}

bool NodeAttrFeature::Match(const Scope *scope) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Failed to invoke Match(), NodeAttrFeature is not properly initialized.");
    return false;
  }

  return impl_->Match(scope);
}

bool ScopeFeature::ScopeFeatureImpl::SubScopesMatch(const std::vector<Scope *> &scopes) {
  int32_t count = 0;
  bool sub_scope_name_matched = false;
  for (auto &scp : scopes) {
    if ((sub_type_.length() > 0UL) && (sub_type_ == scp->SubType())) {
      ++count;
    }
    if (sub_scope_name_matched) {
      continue;
    }
    auto &sub_impl = scp->impl_;
    sub_scope_name_matched = (sub_scope_mask_.length() > 0UL) &&
                             (sub_scope_mask_.length() < scp->Name().length()) &&
                             (sub_impl->LastName().find(sub_scope_mask_) != std::string::npos);
  }

  if ((sub_type_.length() > 0) && (step_ == 0) && (count != num_)) {
    return false;
  }
  if ((sub_scope_mask_.length() > 0UL) && (!sub_scope_name_matched)) {
    return false;
  }

  return true;
}

bool ScopeFeature::ScopeFeatureImpl::Match(const Scope *scope) {
  auto &impl = scope->impl_;
  const std::string scope_name = scope->Name();
  if (suffix_.length() > scope_name.length()) {
    return false;
  }
  if (suffix_.length() > 0UL) {
    const std::string &last_name = impl->LastName();
    if (suffix_ != last_name) {
      return false;
    }
  }

  const std::vector<Scope *> &scopes = impl->GetAllSubScopes();
  if (SubScopesMatch(scopes)) {
    GELOGI("ScopeFeature, match scope:%s", scope->Name().c_str());
    return true;
  }

  return false;
}

ScopeFeature::ScopeFeature(std::string sub_type, int32_t num, std::string suffix,
                           std::string sub_scope_mask, int step)
    : ScopeBaseFeature() {
  impl_ = std::unique_ptr<ScopeFeatureImpl>(new (std::nothrow) ScopeFeatureImpl(sub_type, num, suffix,
                                                                                sub_scope_mask, step));
}

ScopeFeature::ScopeFeature(const char_t *sub_type, int32_t num, const char *suffix,
                           const char_t *sub_scope_mask, int step)
    : ScopeBaseFeature() {
  std::string str_sub_type;
  if (sub_type != nullptr) {
    str_sub_type = sub_type;
  }
  std::string str_suffix;
  if (suffix != nullptr) {
    str_suffix = suffix;
  }
  std::string str_sub_scope_mask;
  if (sub_scope_mask != nullptr) {
    str_sub_scope_mask = sub_scope_mask;
  }
  impl_ = std::unique_ptr<ScopeFeatureImpl>(new (std::nothrow) ScopeFeatureImpl(str_sub_type, num, str_suffix,
                                                                                str_sub_scope_mask, step));
}

ScopeFeature::ScopeFeature(ScopeFeature const &feature) : ScopeBaseFeature() {
  impl_ = std::unique_ptr<ScopeFeatureImpl>(new (std::nothrow) ScopeFeatureImpl(feature.impl_->sub_type_,
                                                                                feature.impl_->num_,
                                                                                feature.impl_->suffix_,
                                                                                feature.impl_->sub_scope_mask_,
                                                                                feature.impl_->step_));
}

ScopeFeature &ScopeFeature::operator=(ScopeFeature const &feature) {
  if (&feature == this) {
    return *this;
  }
  impl_->sub_type_ = feature.impl_->sub_type_;
  impl_->num_ = feature.impl_->num_;
  impl_->suffix_ = feature.impl_->suffix_;
  impl_->sub_scope_mask_ = feature.impl_->sub_scope_mask_;
  impl_->step_ = feature.impl_->step_;
  return *this;
}

ScopeFeature::~ScopeFeature() {}

bool ScopeFeature::Match(const Scope *scope) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Failed to invoke Match(), ScopeFeature is not properly initialized.");
    return false;
  }

  return impl_->Match(scope);
}

bool ScopePattern::ScopePatternImpl::Match(const Scope *scope) const {
  if (scope == nullptr) {
    GELOGE(PARAM_INVALID, "Input scope is nullptr.");
    return false;
  }
  for (auto feature : node_optype_features_) {
    if (!feature.Match(scope)) {
      return false;
    }
  }

  for (auto feature : node_attr_features_) {
    if (!feature.Match(scope)) {
      return false;
    }
  }

  for (auto feature : scopes_features_) {
    if (!feature.Match(scope)) {
      return false;
    }
  }

  // If there is a _Retval node in the scope, the scope will not be fused.
  NodeOpTypeFeature comm_node_feature = NodeOpTypeFeature("_Retval", -1, 0);
  if (!comm_node_feature.Match(scope)) {
    return false;
  }

  return true;
}

void ScopePattern::ScopePatternImpl::SetSubType(const std::string &sub_type) {
  sub_type_ = sub_type;
}

void ScopePattern::ScopePatternImpl::AddNodeOpTypeFeature(NodeOpTypeFeature &feature) {
  node_optype_features_.push_back(feature);
}

void ScopePattern::ScopePatternImpl::AddNodeAttrFeature(NodeAttrFeature &feature) {
  node_attr_features_.push_back(feature);
}

void ScopePattern::ScopePatternImpl::AddScopeFeature(ScopeFeature &feature) {
  scopes_features_.push_back(feature);
}

ScopePattern::ScopePattern() {
  impl_ = std::unique_ptr<ScopePatternImpl>(new (std::nothrow) ScopePatternImpl);
}

ScopePattern::~ScopePattern() {}

ScopePattern &ScopePattern::SetSubType(const std::string &sub_type) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Failed to invoke SetSubType(), ScopePattern is not properly initialized.");
    return *this;
  }
  impl_->SetSubType(sub_type);
  return *this;
}

ScopePattern &ScopePattern::SetSubType(const char_t *sub_type) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Failed to invoke SetSubType(), ScopePattern is not properly initialized.");
    return *this;
  }
  std::string str_sub_type;
  if (sub_type != nullptr) {
    str_sub_type = sub_type;
  }
  impl_->SetSubType(str_sub_type);
  return *this;
}

ScopePattern &ScopePattern::AddNodeOpTypeFeature(NodeOpTypeFeature feature) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Failed to invoke AddNodeOpTypeFeature(), ScopePattern is not properly initialized.");
    return *this;
  }
  impl_->AddNodeOpTypeFeature(feature);
  return *this;
}

ScopePattern &ScopePattern::AddNodeAttrFeature(NodeAttrFeature feature) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Failed to invoke AddNodeAttrFeature(), ScopePattern is not properly initialized.");
    return *this;
  }
  impl_->AddNodeAttrFeature(feature);
  return *this;
}

ScopePattern &ScopePattern::AddScopeFeature(ScopeFeature feature) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Failed to invoke AddScopeFeature(), ScopePattern is not properly initialized.");
    return *this;
  }
  impl_->AddScopeFeature(feature);
  return *this;
}
}  // namespace ge
