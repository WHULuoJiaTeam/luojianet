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

#include "graph/utils/constant_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/op_desc_utils.h"

namespace ge {
bool ConstantUtils::IsConstant(const Operator &op) {
  if (op.GetOpType() == CONSTANT || op.GetOpType() == CONSTANTOP) {
    return true;
  }
  return IsPotentialConst(OpDescUtils::GetOpDescFromOperator(op));
}

bool ConstantUtils::IsConstant(const NodePtr &node) {
  return IsConstant(node->GetOpDesc());
}

bool ConstantUtils::IsConstant(const OpDescPtr &op_desc) {
  if (op_desc->GetType() == CONSTANT || op_desc->GetType() == CONSTANTOP) {
    return true;
  }
  return IsPotentialConst(op_desc);
}

bool ConstantUtils::IsPotentialConst(const OpDescPtr &op_desc) {
  bool is_potential_const = false;
  const auto has_attr = AttrUtils::GetBool(op_desc, ATTR_NAME_POTENTIAL_CONST, is_potential_const);
  return (has_attr && is_potential_const);
}

bool ConstantUtils::IsRealConst(const OpDescPtr &op_desc) {
  return (op_desc->GetType() == CONSTANT || op_desc->GetType() == CONSTANTOP);
}

bool ConstantUtils::GetWeight(const OpDescPtr &op_desc, const uint32_t index, ConstGeTensorPtr &weight) {
  if (AttrUtils::GetTensor(op_desc, ATTR_NAME_WEIGHTS, weight)) {
    return true;
  }
  if (!IsPotentialConst(op_desc)) {
    return false;
  }

  std::vector<uint32_t> weight_indices;
  std::vector<ConstGeTensorPtr> weights;
  if (!GetPotentialWeight(op_desc, weight_indices, weights)) {
    return false;
  }
  for (size_t i = 0U; i < weight_indices.size(); ++i) {
    if (weight_indices[i] == index) {
      weight = weights[i];
      return true;
    }
  }
  return false;
}
bool ConstantUtils::GetWeight(const Operator &op, const uint32_t index, Tensor &weight) {
  if (op.GetAttr(ATTR_NAME_WEIGHTS, weight) == GRAPH_SUCCESS) {
    return true;
  }
  if (!IsPotentialConst(OpDescUtils::GetOpDescFromOperator(op))) {
    return false;
  }
  // check potential const attrs
  std::vector<uint32_t> weight_indices;
  if (op.GetAttr(ATTR_NAME_POTENTIAL_WEIGHT_INDICES, weight_indices) != GRAPH_SUCCESS) {
    GELOGW("Missing ATTR_NAME_POTENTIAL_WEIGHT_INDICES attr on potential const %s.", op.GetName().c_str());
    return false;
  }
  std::vector<Tensor> weights;
  if (op.GetAttr(ATTR_NAME_POTENTIAL_WEIGHT, weights) != GRAPH_SUCCESS) {
    GELOGW("Missing ATTR_NAME_POTENTIAL_WEIGHT attr on potential const %s.", op.GetName().c_str());
    return false;
  }
  if (weight_indices.size() != weights.size()) {
    GELOGW("Weight indices not match with weight size on potential const %s.", op.GetName().c_str());
    return false;
  }

  for (size_t i = 0U; i < weight_indices.size(); ++i) {
    if (weight_indices[i] == index) {
      weight = weights[i];
      return true;
    }
  }
  return false;
}

bool ConstantUtils::MutableWeight(const OpDescPtr &op_desc, const uint32_t index, GeTensorPtr &weight) {
  if (AttrUtils::MutableTensor(op_desc, ATTR_NAME_WEIGHTS, weight)) {
    return true;
  }
  if (!IsPotentialConst(op_desc)) {
    return false;
  }
  std::vector<uint32_t> weight_indices;
  std::vector<GeTensorPtr> weights;
  if (!MutablePotentialWeight(op_desc, weight_indices, weights)) {
    return false;
  }

  for (size_t i = 0U; i < weight_indices.size(); ++i) {
    if (weight_indices[i] == index) {
      weight = weights[i];
      return true;
    }
  }
  return false;
}
bool ConstantUtils::SetWeight(const OpDescPtr &op_desc, const uint32_t index, const GeTensorPtr weight) {
  if (IsRealConst(op_desc) &&
      AttrUtils::SetTensor(op_desc, ATTR_NAME_WEIGHTS, weight)) {
    return true;
  }
  if (!IsPotentialConst(op_desc)) {
    return false;
  }
  std::vector<uint32_t> weight_indices;
  std::vector<GeTensorPtr> weights;
  if (!MutablePotentialWeight(op_desc, weight_indices, weights)) {
    return false;
  }

  for (size_t i = 0U; i < weight_indices.size(); ++i) {
    if (weight_indices[i] == index) {
      weights[i] = weight;
      return AttrUtils::SetListTensor(op_desc, ATTR_NAME_POTENTIAL_WEIGHT, weights);
    }
  }
  return false;
}
bool ConstantUtils::GetPotentialWeight(const OpDescPtr &op_desc,
                                       std::vector<uint32_t> &weight_indices,
                                       std::vector<ConstGeTensorPtr> &weights) {
  // check potential const attrs
  if (!AttrUtils::GetListInt(op_desc, ATTR_NAME_POTENTIAL_WEIGHT_INDICES, weight_indices)) {
    GELOGW("Missing ATTR_NAME_POTENTIAL_WEIGHT_INDICES attr on potential const %s.", op_desc->GetName().c_str());
    return false;
  }
  if (!AttrUtils::GetListTensor(op_desc, ATTR_NAME_POTENTIAL_WEIGHT, weights)) {
    GELOGW("Missing ATTR_NAME_POTENTIAL_WEIGHT attr on potential const %s.", op_desc->GetName().c_str());
    return false;
  }
  if (weight_indices.size() != weights.size()) {
    GELOGW("Weight indices not match with weight size on potential const %s.", op_desc->GetName().c_str());
    return false;
  }
  return true;
}

bool ConstantUtils::MutablePotentialWeight(const OpDescPtr &op_desc, std::vector<uint32_t> &weight_indices,
                                           std::vector<GeTensorPtr> &weights) {
  // check potential const attrs
  if (!AttrUtils::GetListInt(op_desc, ATTR_NAME_POTENTIAL_WEIGHT_INDICES, weight_indices)) {
    GELOGW("Missing ATTR_NAME_POTENTIAL_WEIGHT_INDICES attr on potential const %s.", op_desc->GetName().c_str());
    return false;
  }
  if (!AttrUtils::MutableListTensor(op_desc, ATTR_NAME_POTENTIAL_WEIGHT, weights)) {
    GELOGW("Missing ATTR_NAME_POTENTIAL_WEIGHT attr on potential const %s.", op_desc->GetName().c_str());
    return false;
  }
  if (weight_indices.size() != weights.size()) {
    GELOGW("Weight indices not match with weight size on potential const %s.", op_desc->GetName().c_str());
    return false;
  }
  return true;
}
bool ConstantUtils::MarkPotentialConst(const OpDescPtr &op_desc,
                                       const std::vector<int> indices,
                                       const std::vector<GeTensorPtr> weights) {
  if (indices.size() != weights.size()) {
    return false;
  }
  return (AttrUtils::SetBool(op_desc, ATTR_NAME_POTENTIAL_CONST, true) &&
      AttrUtils::SetListInt(op_desc, ATTR_NAME_POTENTIAL_WEIGHT_INDICES, indices) &&
      AttrUtils::SetListTensor(op_desc, ATTR_NAME_POTENTIAL_WEIGHT, weights));
}
bool ConstantUtils::UnMarkPotentialConst(const OpDescPtr &op_desc) {
  if (op_desc->HasAttr(ATTR_NAME_POTENTIAL_CONST) &&
      op_desc->HasAttr(ATTR_NAME_POTENTIAL_WEIGHT_INDICES) &&
      op_desc->HasAttr(ATTR_NAME_POTENTIAL_WEIGHT)) {
    (void)op_desc->DelAttr(ATTR_NAME_POTENTIAL_CONST);
    (void)op_desc->DelAttr(ATTR_NAME_POTENTIAL_WEIGHT_INDICES);
    (void)op_desc->DelAttr(ATTR_NAME_POTENTIAL_WEIGHT);
    return true;
  }
  return false;
}
} // namespace ge