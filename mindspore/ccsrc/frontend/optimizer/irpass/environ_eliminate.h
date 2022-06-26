/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ENVIRON_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ENVIRON_ELIMINATE_H_

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "utils/hash_map.h"
#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/irpass/inline.h"
#include "frontend/optimizer/optimizer.h"
#include "utils/symbolic.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace internal {
constexpr int kEnvironGetSetInputSize = 4;
constexpr int kEnvironOffset = 1;
constexpr int kSymbolicKeyOffset = 2;
constexpr int kValueOffset = 3;

class EnvironGetTransform {
 public:
  EnvironGetTransform() : cache_() {}
  ~EnvironGetTransform() = default;

  FuncGraphPtr operator()(const FuncGraphPtr &fg, const SymbolicKeyInstancePtr &key, const AnfNodePtr &default_node) {
    if (cache_.find(fg) == cache_.end()) {
      cache_[fg] = {};
    }

    auto &cache = cache_[fg];
    auto hash_key = std::make_pair(key, default_node);
    if (cache.find(hash_key) == cache.end()) {
      std::ostringstream ss("env", std::ostringstream::app);
      if (key->node() != nullptr) {
        ss << key->node()->ToString();
      }

      auto new_fg = TransformableClone(fg, std::make_shared<TraceTransform>(ss.str()));
      auto environ_node = new_fg->output();
      while (IsPrimitiveCNode(environ_node, prim::kPrimEnvironSet)) {
        // {prim::kPrimEnvironSet, environ, symbolickey, value}
        auto &inputs = environ_node->cast<CNodePtr>()->inputs();
        if (inputs.size() != kEnvironGetSetInputSize) {
          MS_LOG(WARNING) << "Input size should be 4";
          return nullptr;
        }
        if (!IsValueNode<SymbolicKeyInstance>(inputs[kSymbolicKeyOffset])) {
          MS_LOG(DEBUG) << "Input 2 is not a SymbolicKeyInstance?";
          return nullptr;
        }

        environ_node = inputs[kEnvironOffset];
        auto value = inputs[kValueOffset];
        auto key2 = GetValueNode<SymbolicKeyInstancePtr>(inputs[kSymbolicKeyOffset]);
        if (*key2 == *key) {
          new_fg->set_output(value);
          cache[hash_key] = new_fg;
          cache_[fg] = cache;
          return new_fg;
        }
      }
      new_fg->set_output(
        new_fg->NewCNode({NewValueNode(prim::kPrimEnvironGet), environ_node, NewValueNode(key), default_node}));
      cache[hash_key] = new_fg;
    }

    return cache[hash_key];
  }

 private:
  mindspore::HashMap<FuncGraphPtr,
                     mindspore::HashMap<std::pair<SymbolicKeyInstancePtr, AnfNodePtr>, FuncGraphPtr, PairHasher>>
    cache_;
};

class EnvironGetTransformACrossGraph {
 public:
  EnvironGetTransformACrossGraph() : cache_() {}
  ~EnvironGetTransformACrossGraph() = default;

  FuncGraphPtr operator()(const FuncGraphPtr &fg, const SymbolicKeyInstancePtr &key, const AnfNodePtr &default_node) {
    if (cache_.find(fg) == cache_.end()) {
      cache_[fg] = {};
    }

    auto &cache = cache_[fg];
    auto hash_key = std::make_pair(key, default_node);
    if (cache.find(hash_key) == cache.end()) {
      std::ostringstream ss("env", std::ostringstream::app);
      if (key->node() != nullptr) {
        ss << key->node()->ToString();
      }

      auto new_fg_outer = TransformableClone(fg, std::make_shared<TraceTransform>(ss.str()));
      auto output_outer = new_fg_outer->output();
      if (!IsValueNode<FuncGraph>(output_outer)) {
        MS_LOG(WARNING) << "Output of outer graph should be a func_graph";
        return nullptr;
      }
      auto fg_inner = GetValueNode<FuncGraphPtr>(output_outer);
      auto new_fg = TransformableClone(fg_inner, std::make_shared<TraceTransform>(ss.str()));
      new_fg_outer->set_output(NewValueNode(new_fg));

      auto environ_node = new_fg->output();
      while (IsPrimitiveCNode(environ_node, prim::kPrimEnvironSet)) {
        // {prim::kPrimEnvironSet, environ, symbolickey, value}
        auto &inputs = environ_node->cast<CNodePtr>()->inputs();
        if (inputs.size() != kEnvironGetSetInputSize) {
          MS_LOG(WARNING) << "Input size should be 4";
          return nullptr;
        }
        if (!IsValueNode<SymbolicKeyInstance>(inputs[kSymbolicKeyOffset])) {
          MS_LOG(DEBUG) << "Input 2 is not a SymbolicKeyInstance?";
          return nullptr;
        }

        environ_node = inputs[kEnvironOffset];
        auto value = inputs[kValueOffset];
        auto key2 = GetValueNode<SymbolicKeyInstancePtr>(inputs[kSymbolicKeyOffset]);
        if (*key2 == *key) {
          new_fg->set_output(value);
          cache[hash_key] = new_fg_outer;
          return new_fg_outer;
        }
      }
      new_fg->set_output(
        new_fg->NewCNode({NewValueNode(prim::kPrimEnvironGet), environ_node, NewValueNode(key), default_node}));
      cache[hash_key] = new_fg_outer;
    }

    return cache[hash_key];
  }

 private:
  mindspore::HashMap<FuncGraphPtr,
                     mindspore::HashMap<std::pair<SymbolicKeyInstancePtr, AnfNodePtr>, FuncGraphPtr, PairHasher>>
    cache_;
};

AnfNodePtr GetIndexedEnvironValueNode(const FuncGraphPtr &fg, const AnfNodePtr &origin_value_node,
                                      const std::size_t index) {
  AnfNodePtr new_value_node;
  if (IsValueNode<ValueTuple>(origin_value_node)) {
    auto origin_value_tuple = GetValueNode<ValueTuplePtr>(origin_value_node);
    if (index >= origin_value_tuple->size()) {
      MS_LOG(EXCEPTION) << "Index: " << index << " is greater than Value size: " << origin_value_tuple->size()
                        << ", Default Value: " << origin_value_node->ToString();
    }
    new_value_node = NewValueNode((*origin_value_tuple)[index]);
  } else {
    new_value_node = fg->NewCNode(
      {NewValueNode(prim::kPrimTupleGetItem), origin_value_node, NewValueNode(MakeValue(static_cast<int64_t>(index)))});
  }
  return new_value_node;
}
}  // namespace internal

// {prim::kPrimEnvironGet, C1, C2, Y} -> Y
class EnvironGetEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    PatternNode c1, c2, y;
    MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimEnvironGet, c1, c2, y), y,
                     (IsNewEnvironNode(c1.GetNode(node)) && IsVNode(c2.GetNode(node))));
    return nullptr;
  }
};

// {prim::kPrimEnvironGet, {prim::kPrimEnvironAdd, X, Y}, C, Z} ->
// {prim::GetPythonOps("hyper_add"), {prim::kPrimEnvironGet, X, C, Z}, {prim::kPrimEnvironGet, Y, C, Z}}
class EnvironGetAddEliminater : public AnfVisitor {
 public:
  EnvironGetAddEliminater() : PrimHyperAdd_(prim::GetPythonOps("hyper_add")) {}
  ~EnvironGetAddEliminater() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    is_match_ = false;
    auto IsAddCNode = [](const AnfNodePtr &node) -> bool {
      return IsPrimitiveCNode(node, prim::kPrimEnvironAdd) && node->cast<CNodePtr>()->size() == 3;
    };
    AnfVisitor::Match(prim::kPrimEnvironGet, {IsAddCNode, IsVNode, IsNode})(node);

    if (!is_match_ || node->func_graph() == nullptr) {
      return nullptr;
    }

    // {prim::kPrimEnvironGet, {...}, C, Z}
    auto cnode = node->cast<CNodePtr>();
    auto inp1 = cnode->input(1)->cast<CNodePtr>();
    auto c = cnode->input(2);
    auto z = cnode->input(3);

    // {prim::kPrimEnvironAdd, X, Y}
    auto x = inp1->input(1);
    auto y = inp1->input(2);

    auto fg = node->func_graph();
    auto xcz = fg->NewCNode({NewValueNode(prim::kPrimEnvironGet), x, c, z});
    auto ycz = fg->NewCNode({NewValueNode(prim::kPrimEnvironGet), y, c, z});

    return fg->NewCNode({NewValueNode(PrimHyperAdd_), xcz, ycz});
  }

  void Visit(const AnfNodePtr &) override { is_match_ = true; }

 private:
  bool is_match_{false};
  ValuePtr PrimHyperAdd_;
};

// {prim::kPrimEnvironGet, {prim::kPrimEnvironSet, X, C1, Y}, C2, Z}
class EnvironGetSetEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    is_match_ = false;
    auto IsSetCNode = [](const AnfNodePtr &node) -> bool {
      if (!IsPrimitiveCNode(node, prim::kPrimEnvironSet)) {
        return false;
      }

      // {prim::kPrimEnvironSet, X, C1, Y}
      auto &inputs = node->cast<CNodePtr>()->inputs();
      if (inputs.size() != 4) {
        return false;
      }

      return IsValueNode<SymbolicKeyInstance>(inputs[2]);
    };
    AnfVisitor::Match(prim::kPrimEnvironGet, {IsSetCNode, IsValueNode<SymbolicKeyInstance>, IsNode})(node);

    if (!is_match_ || node->func_graph() == nullptr) {
      return nullptr;
    }

    // {prim::kPrimEnvironGet, {...}, C2, Z}
    auto cnode = node->cast<CNodePtr>();
    auto inp1 = cnode->input(1)->cast<CNodePtr>();
    auto key2 = cnode->input(2);
    auto c2 = GetValueNode<SymbolicKeyInstancePtr>(key2);
    auto default_v = cnode->input(3);

    // {prim::kPrimEnvironSet, X, C1, Y}
    AnfNodePtr environ_node = inp1->input(1);
    auto c1 = GetValueNode<SymbolicKeyInstancePtr>(inp1->input(2));
    auto last_set = inp1->input(3);

    if (*c1 == *c2) {
      return last_set;
    }

    while (IsPrimitiveCNode(environ_node, prim::kPrimEnvironSet)) {
      // {prim::kPrimEnvironSet, environ, symbolickey, value}
      auto &inputs = environ_node->cast<CNodePtr>()->inputs();
      if (inputs.size() != internal::kEnvironGetSetInputSize) {
        MS_LOG(WARNING) << "Input size should be 4";
        return nullptr;
      }
      if (!IsValueNode<SymbolicKeyInstance>(inputs[internal::kSymbolicKeyOffset])) {
        MS_LOG(DEBUG) << "Input 2 is not a SymbolicKeyInstance?";
        return nullptr;
      }

      environ_node = inputs[internal::kEnvironOffset];
      last_set = inputs[internal::kValueOffset];
      auto symbolic_c1 = GetValueNode<SymbolicKeyInstancePtr>(inputs[internal::kSymbolicKeyOffset]);
      if (*symbolic_c1 == *c2) {
        return last_set;
      }
    }

    return node->func_graph()->NewCNode({NewValueNode(prim::kPrimEnvironGet), environ_node, key2, default_v});
  }

  void Visit(const AnfNodePtr &) override { is_match_ = true; }

 private:
  bool is_match_{false};
};

// {prim::kPrimEnvironGet, {prim::kPrimDepend, X1, X2}, item, dflt} ->
// {prim::kPrimDepend, {prim::kPrimEnvironGet, X1, item, dflt}, X2}
class EnvironGetDependSwap : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!node->isa<CNode>() || node->func_graph() == nullptr) {
      return nullptr;
    }
    ScopePtr scope = node->cast<CNodePtr>()->scope();
    ScopeGuard scope_guard(scope);

    PatternNode x1, x2, item, dflt;
    MATCH_REPLACE(node, PPrimitive(prim::kPrimEnvironGet, PPrimitive(prim::kPrimDepend, x1, x2), item, dflt),
                  PPrimitive(prim::kPrimDepend, PPrimitive(prim::kPrimEnvironGet, x1, item, dflt), x2));
    return nullptr;
  }
};

// {prim::kPrimEnvironAdd, C1, X} -> X
// {prim::kPrimEnvironAdd, X, C1} -> X
class EnvironAddConstEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    PatternNode c1, x;
    MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimEnvironAdd, c1, x), x, (IsNewEnvironNode(c1.GetNode(node))));
    MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimEnvironAdd, x, c1), x, (IsNewEnvironNode(c1.GetNode(node))));
    return nullptr;
  }
};

// {prim::kPrimEnvironGet, {G, Xs}, C, Y}
class IncorporateEnvironGet : public AnfVisitor {
 public:
  explicit IncorporateEnvironGet(bool bypass_recursive = false)
      : environ_get_transform_(), bypass_recursive_(bypass_recursive) {}
  ~IncorporateEnvironGet() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    static bool enable_closure = common::GetEnv("MS_DEV_ENABLE_CLOSURE") != "0";
    if (enable_closure) {
      return nullptr;
    }
    is_match_ = false;
    auto IsGCNode = [](const AnfNodePtr &node) -> bool {
      auto cnode = node->cast<CNodePtr>();
      if (cnode == nullptr || cnode->size() < 1) {
        return false;
      }
      return IsValueNode<FuncGraph>(cnode->input(0));
    };
    AnfVisitor::Match(prim::kPrimEnvironGet, {IsGCNode, IsValueNode<SymbolicKeyInstance>, IsNode})(node);

    if (!is_match_) {
      return nullptr;
    }

    // {prim::kPrimEnvironGet, {...}, C, Y}
    auto cnode = node->cast<CNodePtr>();
    auto inp1 = cnode->input(1)->cast<CNodePtr>();
    auto key = GetValueNode<SymbolicKeyInstancePtr>(cnode->input(2));
    auto default_v = cnode->input(3);

    // {G, Xs}
    auto inputs = inp1->inputs();
    auto fg = GetValueNode<FuncGraphPtr>(inputs[0]);
    auto new_fg = environ_get_transform_(fg, key, default_v);
    if (fg->recursive() && bypass_recursive_) {
      MS_LOG(DEBUG) << "Bypass EnvironGet transform for recursive fg=" << fg->ToString();
      return nullptr;
    }
    if (new_fg == nullptr) {
      return nullptr;
    }
    std::vector<AnfNodePtr> args;
    args.push_back(NewValueNode(new_fg));
    (void)args.insert(args.end(), inputs.begin() + 1, inputs.end());

    return node->func_graph()->NewCNode(args);
  }

  void Visit(const AnfNodePtr &) override { is_match_ = true; }

 private:
  bool is_match_{false};
  internal::EnvironGetTransform environ_get_transform_;
  bool bypass_recursive_;
};

// {prim::kPrimEnvironGet, {{prim::kPrimSwitch, X, G1, G2}, Xs}, C, Y}
class IncorporateEnvironGetSwitch : public AnfVisitor {
 public:
  IncorporateEnvironGetSwitch() : environ_get_transform_() {}
  ~IncorporateEnvironGetSwitch() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    static bool enable_closure = common::GetEnv("MS_DEV_ENABLE_CLOSURE") != "0";
    if (enable_closure) {
      return nullptr;
    }
    is_match_ = false;
    auto IsSwNode = [](const AnfNodePtr &node) -> bool {
      auto cnode = node->cast<CNodePtr>();
      if (cnode == nullptr || cnode->size() < 1) {
        return false;
      }

      return IsPrimitiveCNode(cnode->input(0), prim::kPrimSwitch);
    };
    AnfVisitor::Match(prim::kPrimEnvironGet, {IsSwNode, IsValueNode<SymbolicKeyInstance>, IsNode})(node);
    if (!is_match_ || node->func_graph() == nullptr) {
      return nullptr;
    }

    // {prim::kPrimEnvironGet, {...}, C, Y}
    auto cnode = node->cast<CNodePtr>();
    auto inp1 = cnode->input(1)->cast<CNodePtr>();
    auto key = GetValueNode<SymbolicKeyInstancePtr>(cnode->input(2));
    auto default_v = cnode->input(3);

    // {{prim::kPrimSwitch, X, G1, G2}, Xs}
    auto inputs = inp1->inputs();
    is_match_ = false;
    AnfVisitor::Match(prim::kPrimSwitch, {IsNode, IsValueNode<FuncGraph>, IsValueNode<FuncGraph>})(inputs[0]);
    if (!is_match_) {
      return nullptr;
    }

    // {prim::kPrimSwitch, X, G1, G2}
    auto sw = inputs[0]->cast<CNodePtr>();
    auto x = sw->input(1);
    auto g1 = GetValueNode<FuncGraphPtr>(sw->input(2));
    auto g2 = GetValueNode<FuncGraphPtr>(sw->input(3));
    auto new_g1 = environ_get_transform_(g1, key, default_v);
    auto new_g2 = environ_get_transform_(g2, key, default_v);
    if (new_g1 == nullptr || new_g2 == nullptr) {
      return nullptr;
    }
    auto fg = node->func_graph();
    auto new_sw = fg->NewCNode({NewValueNode(prim::kPrimSwitch), x, NewValueNode(new_g1), NewValueNode(new_g2)});

    std::vector<AnfNodePtr> args{new_sw};
    (void)args.insert(args.end(), inputs.begin() + 1, inputs.end());

    return fg->NewCNode(args);
  }

  void Visit(const AnfNodePtr &) override { is_match_ = true; }

 private:
  bool is_match_{false};
  internal::EnvironGetTransform environ_get_transform_;
};

// {prim::kPrimEnvironGet, {{{prim::kPrimSwitchLayer, X, {prim::kPrimMakeTuple, G1, G2...}}, Xs}, Ys}, C, Y}
class IncorporateEnvironGetSwitchLayer : public AnfVisitor {
 public:
  IncorporateEnvironGetSwitchLayer() : environ_get_transform_() {}
  ~IncorporateEnvironGetSwitchLayer() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    static bool enable_closure = common::GetEnv("MS_DEV_ENABLE_CLOSURE") != "0";
    if (enable_closure) {
      return nullptr;
    }
    is_match_ = false;
    AnfVisitor::Match(prim::kPrimEnvironGet, {IsCNode, IsValueNode<SymbolicKeyInstance>, IsNode})(node);
    if (!is_match_ || node->func_graph() == nullptr) {
      return nullptr;
    }
    // {prim::kPrimEnvironGet, {...}, C, Y}
    auto cnode = node->cast<CNodePtr>();
    auto inp1 = cnode->input(1)->cast<CNodePtr>();
    auto key = GetValueNode<SymbolicKeyInstancePtr>(cnode->input(2));
    auto default_v = cnode->input(3);

    // {{prim::kPrimSwitchLayer, X, {prim::kPrimMakeTuple, G1, G2...}}, Xs}, Ys}
    auto &inputs_outer = inp1->inputs();
    if (!inputs_outer[0]->isa<CNode>()) {
      return nullptr;
    }
    std::vector<AnfNodePtr> args_outer;
    args_outer.insert(args_outer.end(), inputs_outer.begin() + 1, inputs_outer.end());
    auto &input_switch_layer = inputs_outer[0]->cast<CNodePtr>()->inputs();

    is_match_ = false;
    AnfVisitor::Match(prim::kPrimSwitchLayer, {IsNode, IsCNode})(input_switch_layer[0]);
    if (!is_match_) {
      return nullptr;
    }
    std::vector<AnfNodePtr> args;
    (void)args.insert(args.end(), input_switch_layer.begin() + 1, input_switch_layer.end());

    // {prim::kPrimSwitchLayers, X, {prim::kPrimMakeTuple, G1, G2...}}
    auto sw = input_switch_layer[0]->cast<CNodePtr>();
    std::vector<FuncGraphPtr> graphs{};
    auto graphs_cnode = sw->input(2)->cast<CNodePtr>();
    auto &graphs_inputs = graphs_cnode->inputs();
    const int kMinInputSize = 2;
    if (IsPrimitiveCNode(graphs_cnode, prim::kPrimMakeTuple) && graphs_inputs.size() >= kMinInputSize &&
        IsValueNode<FuncGraph>(graphs_inputs[1])) {
      (void)std::transform(graphs_inputs.begin() + 1, graphs_inputs.end(), std::back_inserter(graphs),
                           [](const AnfNodePtr &vnode) { return GetValueNode<FuncGraphPtr>(vnode); });
    }
    if (graphs.empty()) {
      return nullptr;
    }

    auto fg = node->func_graph();
    std::vector<AnfNodePtr> layers;
    for (auto &graph : graphs) {
      auto fg_transform = environ_get_transform_(graph, key, default_v);
      if (fg_transform == nullptr) {
        return nullptr;
      }
      layers.push_back(NewValueNode(fg_transform));
    }
    auto layers_node = fg->NewCNode(prim::kPrimMakeTuple, layers);
    auto new_sw = fg->NewCNode({NewValueNode(prim::kPrimSwitchLayer), sw->input(1), layers_node});
    args.insert(args.begin(), new_sw);
    auto inner_call = fg->NewCNode(args);
    args_outer.insert(args_outer.begin(), inner_call);
    return fg->NewCNode(args_outer);
  }

  void Visit(const AnfNodePtr &) override { is_match_ = true; }

 private:
  bool is_match_{false};
  internal::EnvironGetTransformACrossGraph environ_get_transform_;
};

// {prim::kPrimEnvironSet, E, K, V} ->
//     E1 = {prim::kPrimEnvironSet, E,  K1, V1},
//     E2 = {prim::kPrimEnvironSet, E1, K2, V2},
//     ...
// {prim::kPrimEnvironGet, E, K, V} ->
//     v1 = {prim::kPrimEnvironGet, E, K1, default_v1},
//     v2 = {prim::kPrimEnvironGet, E, K2, devault_v2},
//     ...
//     v_tuple = {prim::kPrimMakeTuple, v1, v2, ...}
class SplitEnvironGetSetWithTupleValue : public AnfVisitor {
 public:
  SplitEnvironGetSetWithTupleValue() = default;
  ~SplitEnvironGetSetWithTupleValue() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!(IsPrimitiveCNode(node, prim::kPrimEnvironSet) || IsPrimitiveCNode(node, prim::kPrimEnvironGet))) {
      return nullptr;
    }
    // {prim::kPrimEnvironSet, E, key, node_with_abstract_is_tuple} or
    // {prim::kPrimEnvironGet, E, key, node_with_abstract_is_tuple}
    const auto &cnode = node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    auto &environ_node = inputs[internal::kEnvironOffset];
    const auto &origin_value_node = inputs[internal::kValueOffset];
    const auto &origin_key_node = GetValueNode<SymbolicKeyInstancePtr>(inputs[internal::kSymbolicKeyOffset]);

    if (origin_key_node == nullptr || origin_value_node->abstract() == nullptr ||
        !origin_value_node->abstract()->isa<abstract::AbstractTuple>()) {
      return nullptr;
    }

    const auto &origin_value_abstract = origin_value_node->abstract()->cast<abstract::AbstractTuplePtr>();
    MS_EXCEPTION_IF_NULL(origin_value_abstract);

    AnfNodePtr prev_environ_node = environ_node;
    auto fg = node->func_graph();

    if (IsPrimitiveCNode(node, prim::kPrimEnvironSet)) {
      CNodePtr new_cnode = cnode;
      // Cascade the split CNode of EnvironSet.
      for (std::size_t index = 0; index < origin_value_abstract->elements().size(); ++index) {
        auto new_key = std::make_shared<SymbolicKeyInstance>(
          origin_key_node->node(), origin_value_abstract->elements()[index], static_cast<int64_t>(index));
        AnfNodePtr new_value_node = internal::GetIndexedEnvironValueNode(fg, origin_value_node, index);

        new_cnode = fg->NewCNode({inputs[0], prev_environ_node, NewValueNode(new_key), new_value_node});
        prev_environ_node = new_cnode;
      }
      return new_cnode;
    } else {
      // MakeTuple the split CNode of EnvironGet.
      AnfNodePtrList tuple_item_list{NewValueNode(prim::kPrimMakeTuple)};
      for (std::size_t index = 0; index < origin_value_abstract->elements().size(); ++index) {
        auto new_key = std::make_shared<SymbolicKeyInstance>(
          origin_key_node->node(), origin_value_abstract->elements()[index], static_cast<int64_t>(index));
        AnfNodePtr new_value_node = internal::GetIndexedEnvironValueNode(fg, origin_value_node, index);
        auto new_item_cnode = fg->NewCNode({inputs[0], environ_node, NewValueNode(new_key), new_value_node});
        tuple_item_list.push_back(new_item_cnode);
      }
      auto new_cnode = fg->NewCNode(tuple_item_list);
      return new_cnode;
    }
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ENVIRON_ELIMINATE_H_
