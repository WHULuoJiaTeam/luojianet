/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/opt.h"

#include <deque>
#include <memory>
#include <algorithm>
#include <utility>

#include "utils/hash_map.h"
#include "ir/anf.h"
#include "ir/manager.h"
#include "frontend/optimizer/optimizer.h"
#include "utils/log_adapter.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {
SubstitutionPtr MakeSubstitution(const OptimizerCallerPtr &transform, const std::string &name, const PrimitivePtr &prim,
                                 const RenormAction &renorm_action, bool has_priority_pattern) {
  auto fn = [prim](const AnfNodePtr &node) -> bool { return IsPrimitiveCNode(node, prim); };
  return std::make_shared<Substitution>(transform, name, fn, renorm_action, has_priority_pattern);
}

SubstitutionPtr MakeSubstitution(const OptimizerCallerPtr &transform, const std::string &name,
                                 const std::vector<PrimitivePtr> &prims, const RenormAction &renorm_action,
                                 bool has_priority_pattern) {
  auto fn = [prims](const AnfNodePtr &node) -> bool {
    if (!node->isa<CNode>()) {
      return false;
    }

    auto cnode = node->cast<CNodePtr>();
    auto inp0 = cnode->input(0);
    auto prim0 = GetValueNode<PrimitivePtr>(inp0);
    if (prim0 == nullptr) {
      return false;
    }

    auto hash = prim0->Hash();
    auto const &name = prim0->name();
    for (auto &prim : prims) {
      if (hash == prim->Hash() && name == prim->name()) {
        return true;
      }
    }
    return false;
  };

  return std::make_shared<Substitution>(transform, name, fn, renorm_action, has_priority_pattern);
}

SubstitutionPtr MakeSubstitution(const OptimizerCallerPtr &transform, const std::string &name,
                                 const PredicateFuncType &predicate, const RenormAction &renorm_action,
                                 bool has_priority_pattern) {
  return std::make_shared<Substitution>(transform, name, predicate, renorm_action, has_priority_pattern);
}

AnfNodePtr Substitution::operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) {
#ifdef ENABLE_PROFILE
  double t = GetTime();
#endif
  AnfNodePtr result = (*transform_)(optimizer, node);
#ifdef ENABLE_PROFILE
  if (optimizer != nullptr) {
    auto time = GetTime();
    MsProfile::StatTime("substitution." + name_, time - t);
    if (result != nullptr) {
      MsProfile::StatTime("match." + name_, time - t);
    }
  }
#endif
  if (optimizer != nullptr && optimizer->is_watch_renormalize() && result != nullptr) {
    if ((renorm_action_ == FORCE_RENORM) || (result->abstract() == nullptr)) {
      optimizer->set_is_untyped_generated();
    }
  }

  return result;
}

static bool isTraversable(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  if (node->isa<CNode>() || node->isa<Parameter>()) {
    return true;
  }
  if (IsValueNode<FuncGraph>(node) || IsValueNode<RefKey>(node)) {
    return true;
  }
  return false;
}

static AnfNodePtr DoTransform(const OptimizerPtr &optimizer, const AnfNodePtr &node,
                              const SubstitutionPtr &substitution) {
  auto manager = optimizer->manager();
  bool is_match = substitution->predicate_(node);
  if (is_match) {
    TraceGuard trace_guard(std::make_shared<TraceOpt>(node->debug_info()));
    ScopeGuard scope_guard(node->scope());
    auto res = (*substitution)(optimizer, node);
    if (res != nullptr && res != node) {
#ifdef ENABLE_PROFILE
      double t = GetTime();
#endif
      MS_LOG(DEBUG) << "Replace " << node->DebugString() << " with " << res->DebugString() << ", by "
                    << substitution->name_;
      (void)manager->Replace(node, res);
#ifdef ENABLE_PROFILE
      MsProfile::StatTime("replace." + substitution->name_, GetTime() - t);
#endif
      return res;
    }
  }
  return nullptr;
}

static void UpdateTransformingListForSubstitutions(const AnfNodePtr &node, std::deque<AnfNodePtr> *todo, bool change) {
  if (IsValueNode<FuncGraph>(node)) {
    (*todo).emplace_back(GetValueNode<FuncGraphPtr>(node)->output());
  }

  if (change) {
    (*todo).emplace_back(node);
  } else {
    if (node->isa<CNode>()) {
      auto &inputs = node->cast<CNodePtr>()->inputs();
      (void)std::copy(inputs.begin(), inputs.end(), std::back_inserter(*todo));
    }
  }
}

static void UpdateTransformingListForIR(const AnfNodePtr &node, std::deque<AnfNodePtr> *todo, bool change,
                                        const SubstitutionPtr &substitution) {
  if (IsValueNode<FuncGraph>(node)) {
    (*todo).emplace_back(GetValueNode<FuncGraphPtr>(node)->output());
  }

  // If there is a priority pattern in substitution, don't transform the new node,
  // otherwise some nodes may match the wrong patterns.
  if (change && substitution != nullptr && !substitution->has_priority_pattern_) {
    (*todo).emplace_back(node);
  } else {
    if (node->isa<CNode>()) {
      auto &inputs = node->cast<CNodePtr>()->inputs();
      (void)std::copy(inputs.begin(), inputs.end(), std::back_inserter(*todo));
    }
  }
}

static void UpdateTransformingListWithUserNodes(const OptimizerPtr &optimizer, const AnfNodePtr &node,
                                                std::deque<AnfNodePtr> *todo, bool change, SeenNum seen) {
  if (!change) {
    return;
  }
  auto manager = optimizer->manager();
  auto &node_users = manager->node_users();
  auto users_iterator = node_users.find(node);
  if (users_iterator == node_users.end()) {
    return;
  }
  auto users = users_iterator->second;
  for (auto &use : users) {
    auto use_node = use.first;
    if (use_node == nullptr) {
      continue;
    }
    (*todo).emplace_back(use_node);
    if (use_node->seen_ == seen) {
      use_node->seen_--;
    }
  }
}

bool SubstitutionList::ApplyIRToSubstitutions(const OptimizerPtr &optimizer, const FuncGraphPtr &func_graph) const {
#ifdef ENABLE_PROFILE
  double start = GetTime();
#endif
  FuncGraphManagerPtr manager = optimizer->manager();
  auto seen = NewSeenGeneration();
  std::deque<AnfNodePtr> todo;
  todo.emplace_back(func_graph->output());
  bool changes = false;

  auto &all_nodes = manager->all_nodes();
  while (!todo.empty()) {
    AnfNodePtr node = todo.front();
    todo.pop_front();

    if (node == nullptr || node->seen_ == seen || !isTraversable(node) || !all_nodes.contains(node)) {
      continue;
    }
    node->seen_ = seen;

    bool change = false;
    for (auto &substitution : list_) {
      auto res = DoTransform(optimizer, node, substitution);
      if (res != nullptr) {
        change = true;
        changes = true;
        node = res;
        break;
      }
    }
    UpdateTransformingListForSubstitutions(node, &todo, change);
    UpdateTransformingListWithUserNodes(optimizer, node, &todo, change, seen);
  }
#ifdef ENABLE_PROFILE
  MsProfile::StatTime("opt.transforms." + optimizer->name(), GetTime() - start);
#endif
  return changes;
}

bool SubstitutionList::ApplySubstitutionToIR(const OptimizerPtr &optimizer, const FuncGraphPtr &func_graph,
                                             const SubstitutionPtr &substitution) const {
#ifdef ENABLE_PROFILE
  double start = GetTime();
#endif
  FuncGraphManagerPtr manager = optimizer->manager();
  auto seen = NewSeenGeneration();
  std::deque<AnfNodePtr> todo;
  todo.emplace_back(func_graph->output());
  bool changes = false;

  auto &all_nodes = manager->all_nodes();
  while (!todo.empty()) {
    AnfNodePtr node = todo.front();
    todo.pop_front();

    if (node == nullptr || node->seen_ == seen || !isTraversable(node) || !all_nodes.contains(node)) {
      continue;
    }
    node->seen_ = seen;

    bool change = false;
    auto res = DoTransform(optimizer, node, substitution);
    if (res != nullptr) {
      change = true;
      changes = true;
      node = res;
    }
    UpdateTransformingListForIR(node, &todo, change, substitution);
    UpdateTransformingListWithUserNodes(optimizer, node, &todo, change, seen);
  }

#ifdef ENABLE_PROFILE
  MsProfile::StatTime("opt.transform." + optimizer->name(), GetTime() - start);
#endif
  return changes;
}

void SubstitutionList::DisplayStatusOfSubstitution(const mindspore::HashMap<std::string, std::vector<bool>> &status,
                                                   const OptimizerPtr &optimizer, size_t space) const {
  constexpr int pad_width = 4;
  std::stringstream ss;
  ss << std::endl
     << "Pass: " << optimizer->name() << "(" << optimizer->CurPass_.counter << ")_" << optimizer->CurPass_.name
     << std::endl;
  for (size_t i = 0; i < list_.size(); i++) {
    auto name = list_[i]->name_;
    ss << std::left << std::setw(SizeToInt(space) + pad_width) << name << "\t";
    for (auto change : status.at(name + std::to_string(i))) {
      ss << change << " ";
    }
    ss << std::endl;
  }
  MS_LOG(DEBUG) << ss.str();
}

bool SubstitutionList::ApplySubstitutionsToIR(const OptimizerPtr &optimizer, const FuncGraphPtr &func_graph) const {
  // Add for substitution status counting
  size_t space = 0;
  mindspore::HashMap<std::string, std::vector<bool>> status;
  if (optimizer->is_on_debug_) {
    for (size_t i = 0; i < list_.size(); i++) {
      status[list_[i]->name_ + std::to_string(i)] = {};
    }
  }

  bool changes = false;
  bool loop = true;
  while (loop) {
    loop = false;
    for (size_t i = 0; i < list_.size(); i++) {
      const auto &substitution = list_[i];
      bool change = ApplySubstitutionToIR(optimizer, func_graph, substitution);
      changes = changes || change;
      loop = loop || change;
#ifdef ENABLE_DUMP_IR
      static const auto enable_dump_pass_ir = GetDumpConfig().enable_dump_pass_ir;
      if (enable_dump_pass_ir && MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG)) {
        auto fg_name = optimizer->name() + "_r" + std::to_string(optimizer->CurPass_.counter) + "_" +
                       optimizer->CurPass_.name + "_" + substitution->name_;
        DumpIR(fg_name + ".ir", func_graph);
        if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode) {
          ExportIR(fg_name + ".dat", func_graph);
          draw::Draw(fg_name + ".dot", func_graph);
        }
      }
#endif

      // Record the status of each substitution
      if (optimizer->is_on_debug_) {
        status[substitution->name_ + std::to_string(i)].push_back(change);
        space = std::max(substitution->name_.size(), space);
      }
    }
    if (is_once_) {
      break;
    }
  }

  // Display the status of each substitution
  if (optimizer->is_on_debug_) {
    DisplayStatusOfSubstitution(status, optimizer, space);
  }
  return changes;
}

bool SubstitutionList::operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) const {
  MS_EXCEPTION_IF_NULL(optimizer);
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = optimizer->manager();
  manager->AddFuncGraph(func_graph);
  bool changes = false;
  static const auto traverse_mode =
    (common::GetEnv("MS_DEV_TRAVERSE_SUBSTITUTIONS_MODE") != "1" ? kOptTraverseFromIRToSubstitutions
                                                                 : kOptTraverseFromSubstitutionsToIR);
  if (traverse_mode == kOptTraverseFromIRToSubstitutions &&
      MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode &&
      optimizer->traverse_nodes_first() && !is_once_ && !global_sensitive_) {
    MS_LOG(DEBUG) << "IR >> SUB, " << optimizer->name() << "(r" << optimizer->CurPass_.counter << ")_"
                  << optimizer->CurPass_.name;
    changes = ApplyIRToSubstitutions(optimizer, func_graph);
  } else {
    MS_LOG(DEBUG) << "SUB >> IR, " << optimizer->name() << "(r" << optimizer->CurPass_.counter << ")_"
                  << optimizer->CurPass_.name;
    changes = ApplySubstitutionsToIR(optimizer, func_graph);
  }
  return changes;
}

bool SimpleRewriter::Run() {
  bool changed = false;
  auto seen = NewSeenGeneration();
  std::deque<AnfNodePtr> todo;
  auto add_todo = [&seen, &todo](const AnfNodePtr &node) {
    if (node != nullptr && node->seen_ != seen) {
      (void)todo.emplace_back(node);
    }
  };
  (void)todo.emplace_back(root_graph_->output());
  auto &all_nodes = manager_->all_nodes();
  while (!todo.empty()) {
    AnfNodePtr node = std::move(todo.front());
    todo.pop_front();
    if (node == nullptr || node->seen_ == seen || !all_nodes.contains(node)) {
      continue;
    }
    node->seen_ = seen;
    auto cnode = node->cast<CNodePtr>();
    if (cnode != nullptr) {
      for (auto &input : cnode->inputs()) {
        add_todo(input);
      }
    } else {
      auto fg = GetValueNode<FuncGraphPtr>(node);
      if (fg != nullptr) {
        add_todo(fg->output());
      }
    }
    auto new_node = NodeRewrite(node);
    if (new_node != nullptr) {
      manager_->Replace(node, new_node);
      changed = true;
    }
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
