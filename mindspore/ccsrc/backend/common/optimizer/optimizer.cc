/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "backend/common/optimizer/optimizer.h"

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "backend/common/optimizer/pass_manager.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/manager.h"

namespace mindspore {
namespace opt {
PatternProcessPass::PatternProcessPass(const std::string &name, bool multigraph)
    : NodePass(name),
      multigraph_(multigraph),
      pattern_engine_(PatternEngine(std::make_shared<Visitor>())),
      primitive_vars_(std::make_shared<PrimitiveVarMap>()),
      equiv_(std::make_shared<Equiv>()) {}

const BaseRef PatternProcessPass::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  return BaseRef({X});
}

void PatternProcessPass::Build() {
  VarPtr fg = std::make_shared<Var>("RootG");
  pattern_ = SexpToNode(DefinePattern(), fg, primitive_vars_.get(), multigraph_);
}

AnfNodePtr PatternProcessPass::Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  if (pattern_ == nullptr) {
    Build();
  }

  auto primitive = GetCNodePrimitive(pattern_);
  if (IsPrimitiveCNode(node, primitive)) {
    MS_EXCEPTION_IF_NULL(primitive_vars_);
    MS_EXCEPTION_IF_NULL(equiv_);
    equiv_->clear();
    EquivPtr equiv = pattern_engine_.Match(pattern_, node, *primitive_vars_, equiv_);
    if (equiv != nullptr && !equiv->empty()) {
      return Process(func_graph, node, equiv);
    }
  }
  return nullptr;
}

std::vector<AnfNodePtr> PatternProcessPass::GetOrigNodes() const {
  std::vector<AnfNodePtr> orig_nodes;
  for (auto &prim_var : *primitive_vars_) {
    if (equiv_->find(prim_var.second) == equiv_->end()) {
      continue;
    }
    auto baseref = (*equiv_)[prim_var.second];
    if (!utils::isa<CNode>(baseref)) {
      continue;
    }
    auto node = utils::cast<AnfNodePtr>(baseref);
    orig_nodes.push_back(node);
  }
  return orig_nodes;
}

CNodePtr PatternProcessPass::NewCNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &fg) const {
  MS_EXCEPTION_IF_NULL(fg);
  auto orig_nodes = GetOrigNodes();
  return opt::NewCNode(inputs, fg, orig_nodes);
}

CNodePtr PatternProcessPass::NewCNode(const CNodePtr &cnode, const KernelGraphPtr &fg) const {
  MS_EXCEPTION_IF_NULL(fg);
  auto orig_nodes = GetOrigNodes();
  return opt::NewCNode(cnode, fg, orig_nodes);
}

bool MultipleOutputPatternProcessPass::MatchAnotherPattern(const AnfNodePtr &node, const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  VarPtr fg = std::make_shared<Var>("RootG");
  MS_EXCEPTION_IF_NULL(child_primitive_vars_);
  MS_EXCEPTION_IF_NULL(child_equiv_);
  EquivPtr another_equiv =
    child_pattern_engine_.Match(SexpToNode(DefineAnotherPattern(), fg, child_primitive_vars_.get(), true), node,
                                *child_primitive_vars_, child_equiv_);
  if (another_equiv != nullptr && !another_equiv->empty()) {
    return IsShareNodes(equiv, another_equiv);
  }
  return false;
}

std::vector<AnfNodePtr> MultipleOutputPatternProcessPass::GetOrigNodes() const {
  std::vector<AnfNodePtr> orig_nodes = PatternProcessPass::GetOrigNodes();
  for (auto &prim_var : *child_primitive_vars_) {
    auto baseref = (*child_equiv_)[prim_var.second];
    if (!utils::isa<CNode>(baseref)) {
      continue;
    }
    auto node = utils::cast<AnfNodePtr>(baseref);
    orig_nodes.push_back(node);
  }
  return orig_nodes;
}

void GraphOptimizer::AddPassManager(const PassManagerPtr &pass_manager) {
  if (pass_manager != nullptr) {
    pass_managers_.push_back(pass_manager);
  }
}

FuncGraphPtr GraphOptimizer::Optimize(const FuncGraphPtr &func_graph, bool run_only_once) {
  MS_EXCEPTION_IF_NULL(func_graph);
  run_only_once_ = (pass_managers_.size() == 1) ? true : run_only_once;
  // cppcheck-suppress *
  auto manager = Manage(func_graph, true);

  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < pass_managers_.size(); ++i) {
      const PassManagerPtr &pm = pass_managers_[i];
      if (pm != nullptr && pm->Run(func_graph)) {
        changed = true;
      }
    }
    if (run_only_once_) {
      break;
    }
  }

  std::vector<FuncGraphPtr> func_graphs;
  func_graphs.push_back(func_graph);
  (void)TopoSort(func_graph->get_return());
  return func_graph;
}
}  // namespace opt
}  // namespace mindspore
