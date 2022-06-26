/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "include/common/debug/anf_dump_utils.h"
#include "abstract/abstract_function.h"

namespace mindspore {
namespace {
std::string GetAbstractFuncStr(const abstract::AbstractFunctionPtr &abs) {
  std::ostringstream oss;
  if (abs->isa<abstract::FuncGraphAbstractClosure>()) {
    const auto &abstract_func_graph = abs->cast<abstract::FuncGraphAbstractClosurePtr>();
    if (abstract_func_graph->func_graph() != nullptr) {
      oss << abstract_func_graph->func_graph()->ToString();
    }
  }
  if (abs->isa<abstract::PartialAbstractClosure>()) {
    const auto &abstract_partial_func = abs->cast<abstract::PartialAbstractClosurePtr>();
    const auto &abstract_fn = abstract_partial_func->fn();
    if (abstract_fn->isa<abstract::FuncGraphAbstractClosure>()) {
      const auto &abstract_func_graph = abstract_fn->cast<abstract::FuncGraphAbstractClosurePtr>();
      if (abstract_func_graph->func_graph() != nullptr) {
        oss << "Partial(" << abstract_func_graph->func_graph()->ToString() << ")";
      }
    }
  }
  return oss.str();
}
}  // namespace

std::string GetNodeFuncStr(const AnfNodePtr &nd) {
  MS_EXCEPTION_IF_NULL(nd);
  std::ostringstream oss;
  std::string str;
  const auto &abs = nd->abstract();

  if (IsValueNode<FuncGraph>(nd) || abs == nullptr || !abs->isa<abstract::AbstractFunction>()) {
    return str;
  }
  const auto &abs_func = abs->cast<abstract::AbstractFunctionPtr>();

  if (abs_func->isa<abstract::AbstractFuncUnion>()) {
    oss << "FuncUnion(";
    bool first_item = true;
    auto build_oss = [&oss, &first_item](const abstract::AbstractFuncAtomPtr &poss) {
      auto abs_str = GetAbstractFuncStr(poss);
      if (!first_item) {
        oss << ", ";
      } else {
        first_item = false;
      }
      if (!abs_str.empty()) {
        oss << abs_str;
      }
    };
    abs_func->Visit(build_oss);
    oss << ")";
    return oss.str();
  }
  return GetAbstractFuncStr(abs_func);
}

std::string TypeToShortString(const TypeId &typeId) {
  std::string label = TypeIdLabel(typeId);
  std::string prefix = "kNumberType";
  if (prefix.length() > label.length()) {
    return label;
  }
  auto position = label.find(prefix);
  // Position is 0 when label begins with prefix
  if (position != 0) {
    return label;
  }
  auto sub_position = position + prefix.length();
  if (sub_position >= label.length()) {
    return label;
  }
  return label.substr(sub_position);
}

std::string GetKernelNodeName(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::string kernel_name = anf_node->fullname_with_scope();
  if (kernel_name.empty()) {
    kernel_name = anf_node->ToString();
  }
  MS_LOG(DEBUG) << "Full scope kernel name is " << kernel_name << ".";
  return kernel_name;
}
}  // namespace mindspore
