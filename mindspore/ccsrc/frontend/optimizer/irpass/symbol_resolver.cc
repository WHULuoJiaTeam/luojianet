/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/irpass/symbol_resolver.h"

#include <string>
#include <memory>
#include <vector>

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimGetAttr, object, attr}
// {prim::kPrimResolve, namespace, symbol}
AnfNodePtr Resolver::operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) {
  PatternNode<AnfNodePtr> object, attr, ns_node, sym_node;
  auto GetAttrLambda = [&node, &object, &attr, &optimizer]() -> AnfNodePtr {
    auto object_node = object.GetNode(node);
    auto attr_node = attr.GetNode(node);

    // {prim::kPrimGetAttr, {prim::kPrimResolve, namespace, symbol}, attr}
    if (IsPrimitiveCNode(object_node, prim::kPrimResolve)) {
      auto [name_space, symbol] = parse::GetNamespaceAndSymbol(object_node);
      auto module_name = name_space->module();
      constexpr std::string_view parse_super_name = "namespace";
      if (module_name.find(parse::RESOLVE_NAMESPACE_NAME_CLASS_MEMBER) != std::string::npos &&
          symbol->symbol() != parse_super_name) {
        auto symbol_obj = parse::GetSymbolObject(name_space, symbol, node);
        return parse::ResolveCellWithAttr(optimizer->manager(), symbol_obj, object_node, attr_node);
      }
    }

    // {prim::kPrimGetAttr, {getitem, {prim::kPrimResolve, namespace, symbol}, index}, attr}
    if (parse::IsGetItemCNode(object_node)) {
      auto getitem_cnode = object_node->cast<CNodePtr>();
      constexpr auto resolve_index = 1;
      constexpr auto index_index = 2;
      auto resolve_node = getitem_cnode->input(resolve_index);
      auto index_node = getitem_cnode->input(index_index);
      if (IsPrimitiveCNode(resolve_node, prim::kPrimResolve)) {
        auto [name_space, symbol] = parse::GetNamespaceAndSymbol(resolve_node);
        auto obj = parse::GetObjectFromSequence(name_space, symbol, resolve_node, index_node);
        if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
          return parse::ResolveSequenceWithAttr(optimizer->manager(), obj, resolve_node, attr_node, getitem_cnode);
        }
        return parse::ResolveCellWithAttr(optimizer->manager(), obj, resolve_node, attr_node);
      }
    }

    // {prim::kPrimGetAttr, namespace, attr}
    if (IsValueNode<parse::NameSpace>(object_node)) {
      auto name_space = GetValueNode<parse::NameSpacePtr>(object_node);
      auto attr_str = GetValue<std::string>(GetValueNode(attr_node));
      parse::SymbolPtr symbol = std::make_shared<parse::Symbol>(attr_str);
      return parse::ResolveSymbol(optimizer->manager(), name_space, symbol, node);
    }

    // {prim::kPrimGetAttr, MsClassObject, attr}
    if (IsValueNode<parse::MsClassObject>(object_node)) {
      auto ms_class = GetValueNode<parse::MsClassObjectPtr>(object_node);
      auto attr_str = GetValue<std::string>(GetValueNode(attr_node));
      return parse::ResolveMsClassWithAttr(optimizer->manager(), ms_class, attr_str, node);
    }

    // {prim::kPrimGetAttr, bool, attr}
    if (IsValueNode<BoolImm>(object_node)) {
      return object_node;
    }
    return nullptr;
  };

  auto ResolveLambda = [&node, &ns_node, &sym_node, &optimizer]() -> AnfNodePtr {
    auto name_space = GetValueNode<parse::NameSpacePtr>(ns_node.GetNode(node));
    auto symbol = GetValueNode<parse::SymbolPtr>(sym_node.GetNode(node));
    auto manager = optimizer->manager();
    return parse::ResolveSymbol(manager, name_space, symbol, node);
  };

  // {prim::kPrimGetAttr, object, attr}
  MATCH_REPLACE_LAMBDA_IF(node, PPrimitive(prim::kPrimGetAttr, object, attr), GetAttrLambda,
                          attr.CheckFunc(IsValueNode<StringImm>, node));
  // {prim::kPrimResolve, namespace, symbol}
  MATCH_REPLACE_LAMBDA_IF(
    node, PPrimitive(prim::kPrimResolve, ns_node, sym_node), ResolveLambda,
    ns_node.CheckFunc(IsValueNode<parse::NameSpace>, node) && sym_node.CheckFunc(IsValueNode<parse::Symbol>, node));
  return nullptr;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
