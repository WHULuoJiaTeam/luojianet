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

#ifndef REGISTER_SCOPE_SCOPE_PATTERN_IMPL_H_
#define REGISTER_SCOPE_SCOPE_PATTERN_IMPL_H_

#include <stdexcept>
#include <limits>
#include <cmath>
#include "external/register/scope/scope_fusion_pass_register.h"
#include "external/graph/types.h"
#include "graph/compute_graph.h"

namespace ge {
class ScopeAttrValue::ScopeAttrValueImpl {
 public:
  ScopeAttrValueImpl() : int_value_(0), float_value_(0.0F), string_value_(""), bool_value_(false) {}
  ~ScopeAttrValueImpl() {}

  void SetIntValue(const int64_t &value) { int_value_ = value; }
  void SetFloatValue(const float32_t &value) { float_value_ = value; }
  void SetStringValue(const std::string &value) { string_value_ = value; }
  void SetBoolValue(const bool &value) { bool_value_ = value; }
  const int64_t &GetIntValue() const { return int_value_; }
  const float32_t &GetFloatValue() const { return float_value_; }
  const std::string &GetStrValue() const { return string_value_; }
  const bool &GetBoolValue() const { return bool_value_; }

 private:
  int64_t int_value_;
  float32_t float_value_;
  std::string string_value_;
  bool bool_value_;
};


class NodeOpTypeFeature::NodeOpTypeFeatureImpl : ScopeBaseFeature {
 public:
  NodeOpTypeFeatureImpl(const std::string nodeType, const int64_t num, const int64_t step = 0)
      : ScopeBaseFeature(), node_type_(nodeType), num_(num), step_(step) {}
  ~NodeOpTypeFeatureImpl() override {}
  bool Match(const Scope *const scope) override;

private:
  std::string node_type_;  // Node type
  int64_t num_;           // Node number
  int64_t step_;          // step
  friend class NodeOpTypeFeature;
};

class NodeAttrFeature::NodeAttrFeatureImpl : ScopeBaseFeature {
 public:
  NodeAttrFeatureImpl(const std::string nodeType, const std::string attr_name, const ge::DataType datatype,
                      const ScopeAttrValue &attr_value)
      : ScopeBaseFeature(), node_type_(nodeType), attr_name_(attr_name), datatype_(datatype),
        attr_value_(attr_value) {}
  ~NodeAttrFeatureImpl() override {}
  bool Match(const Scope *scope) override;
  Status CheckNodeAttrFeatureData(const bool init_value, const ge::OpDescPtr &op_desc, const Scope *const scope);
  Status CheckNodeAttrFeatureData(const std::string init_value, const ge::OpDescPtr &op_desc, const Scope *const scope);
  Status CheckNodeAttrFeatureData(const int64_t init_value, const ge::OpDescPtr &op_desc, const Scope *const scope);
  Status CheckNodeAttrFeatureData(const float32_t init_value, const ge::OpDescPtr &op_desc, const Scope *const scope);
  template<class T>
  typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type FloatIsEqual(const T x, const T y) const
  {
    // It is used for floating point comparisons.
    // It mainly uses relative precision to judge whether floating-point numbers are equal.
    // the 2 is ULPs
    return (std::fabs(x - y) <= (std::numeric_limits<T>::epsilon() * std::fabs(x + y) * 2.0)) ||
           (std::fabs(x - y) < std::numeric_limits<T>::min());
  }

private:
  std::string node_type_;                        // Node type
  std::string attr_name_;                        // attribute name
  ge::DataType datatype_;     // datatype
  ScopeAttrValue attr_value_;  // AttrValue
  friend class NodeAttrFeature;
};

class ScopeFeature::ScopeFeatureImpl : ScopeBaseFeature {
 public:
  ScopeFeatureImpl(const std::string sub_type, const int32_t num, const std::string suffix = "",
                   const std::string sub_scope_mask = "", const int64_t step = 0)
      : ScopeBaseFeature(), sub_type_(sub_type), num_(num), suffix_(suffix), sub_scope_mask_(sub_scope_mask),
        step_(step) {}
  ~ScopeFeatureImpl() override {}
  bool Match(const Scope *const scope) override;
  bool SubScopesMatch(const std::vector<Scope *> &scopes);

 private:
  std::string sub_type_;
  int32_t num_;
  std::string suffix_;
  std::string sub_scope_mask_;
  int64_t step_;
  friend class ScopeFeature;
};

class ScopePattern::ScopePatternImpl {
 public:
  ScopePatternImpl() {}
  ~ScopePatternImpl() {}
  bool Match(const Scope *scope) const;
  void SetSubType(const std::string &sub_type);
  const std::string &SubType() const { return sub_type_; }
  void AddNodeOpTypeFeature(NodeOpTypeFeature &feature);
  void AddNodeAttrFeature(NodeAttrFeature &feature);
  void AddScopeFeature(ScopeFeature &feature);

 private:
  std::string sub_type_;  // get Scope sub type
  std::vector<NodeOpTypeFeature> node_optype_features_;
  std::vector<NodeAttrFeature> node_attr_features_;
  std::vector<ScopeFeature> scopes_features_;
};
}  // namespace ge
#endif  // REGISTER_SCOPE_SCOPE_PATTERN_IMPL_H_