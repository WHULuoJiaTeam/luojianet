/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_PARAMETER_ELIMINATE_H
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_PARAMETER_ELIMINATE_H
#include <vector>
#include <utility>
#include <memory>

#include "utils/hash_set.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "ir/manager.h"
#include "ir/func_graph.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
static inline std::vector<CNodePtr> GetCallers(const FuncGraphPtr &fg) {
  const auto &fg_caller_and_indexes = fg->func_graph_cnodes_index();
  std::vector<CNodePtr> caller_cnodes = {};
  // Find all caller of fg.
  for (const auto &it : fg_caller_and_indexes) {
    const auto &fg_caller_and_index = it.first;
    auto caller_cnode = fg_caller_and_index->first;
    auto index = fg_caller_and_index->second;
    // If index != 0, the caller is a indirect caller, can't erase the parameter of graph.Because
    // in this situation ValueNode<FuncGraph> is a input of Return or of MakeTuple.
    if (index != 0) {
      return {};
    }
    (void)caller_cnodes.emplace_back(caller_cnode->cast<CNodePtr>());
  }
  return caller_cnodes;
}

static inline std::pair<FuncGraphPtr, std::vector<CNodePtr>> SearchFuncGraphCallers(
  const FuncGraphPtr &func_graph, bool eliminate_only_returned_parameter) {
  for (const auto &fg : func_graph->func_graphs_used_total()) {
    if (fg->has_flag(FUNC_GRAPH_FLAG_DEFER_INLINE)) {
      continue;
    }
    const auto &parameters = fg->parameters();
    MS_EXCEPTION_IF_NULL(fg->manager());
    const auto &manager_node_users = fg->manager()->node_users();
    // Check if no user parameter or only one user in output tuple.
    bool exist_param_unused =
      std::any_of(parameters.begin(), parameters.end(),
                  [&manager_node_users, &fg, eliminate_only_returned_parameter](const AnfNodePtr &parameter) {
                    const auto &node_users_it = manager_node_users.find(parameter);
                    // No user parameter.
                    if (node_users_it == manager_node_users.end() || node_users_it->second.empty()) {
                      return true;
                    }
                    // We will check the tuple output, if only one user.
                    if (eliminate_only_returned_parameter && fg->has_flag(FUNC_GRAPH_FLAG_NO_INLINE) &&
                        node_users_it->second.size() == 1) {
                      auto user = node_users_it->second.begin()->first;
                      // The parameter only used as returned MakeTuple's element.
                      if (IsPrimitiveCNode(user, prim::kPrimMakeTuple) && fg->output() == user) {
                        return true;
                      }
                    }
                    return false;
                  });
    if (exist_param_unused) {
      const auto &callers = GetCallers(fg);
      if (!callers.empty()) {
        return {fg, callers};
      }
    }
  }
  return {nullptr, {}};
}

static inline std::pair<mindspore::HashSet<size_t>, mindspore::HashMap<size_t, size_t>> EraseUnusedParameters(
  const FuncGraphPtr &fg, bool eliminate_only_returned_parameter) {
  const FuncGraphManagerPtr &manager = fg->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &manager_node_users = manager->node_users();
  const auto &parameters = fg->parameters();
  mindspore::HashSet<size_t> unused_parameter_indexes;
  mindspore::HashMap<size_t, size_t> only_return_parameter_indexes;
  // Traverse to find all unused parameters.
  size_t index = 0;
  for (const auto &parameter : parameters) {
    const auto &node_users_it = manager_node_users.find(parameter);
    if (node_users_it == manager_node_users.end() || node_users_it->second.empty()) {
      (void)unused_parameter_indexes.emplace(index);
    } else if (eliminate_only_returned_parameter && fg->has_flag(FUNC_GRAPH_FLAG_NO_INLINE) &&
               node_users_it->second.size() == 1) {
      auto user = node_users_it->second.begin()->first;
      auto pos = node_users_it->second.begin()->second;
      // The parameter only used as returned MakeTuple's element.
      if (IsPrimitiveCNode(user, prim::kPrimMakeTuple) && fg->output() == user) {
        MS_LOG(DEBUG) << "Found only returned parameter[" << index << "] at output index[" << pos << "] of "
                      << user->DebugString();
        (void)only_return_parameter_indexes.emplace(pos, index);
        (void)unused_parameter_indexes.emplace(index);
        // Erase the unused element in returned MakeTuple CNode.
        auto user_cnode = dyn_cast<CNode>(user);
        MS_EXCEPTION_IF_NULL(user_cnode);
        auto zero_value = NewValueNode(MakeValue(0));
        zero_value->set_abstract(std::make_shared<abstract::AbstractScalar>(std::make_shared<Int32Imm>(0)));
        user_cnode->set_input(pos, zero_value);
      }
    }
    index++;
  }
  // Erase unused parameters.
  std::vector<AnfNodePtr> new_parameters;
  for (size_t i = 0; i < parameters.size(); i++) {
    if (unused_parameter_indexes.find(i) == unused_parameter_indexes.end()) {
      (void)new_parameters.emplace_back(parameters[i]);
    } else {
      MS_LOG(DEBUG) << "Erase parameter:" << parameters[i]->DebugString() << ",index:" << i;
    }
  }
  manager->SetParameters(fg, new_parameters);
  return {unused_parameter_indexes, only_return_parameter_indexes};
}

// Adjust the call arguments of func graph whose parameter's eliminated.
static inline void AdjustCallerArgs(const CNodePtr &caller,
                                    const mindspore::HashSet<size_t> &unused_parameter_indexes) {
  const FuncGraphManagerPtr &manager = caller->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<AnfNodePtr> new_args = {caller->input(0)};
  for (size_t i = 0; i < caller->inputs().size() - 1; i++) {
    if (unused_parameter_indexes.find(i) == unused_parameter_indexes.end()) {
      (void)new_args.emplace_back(caller->inputs()[i + 1]);
    } else {
      MS_LOG(DEBUG) << "Erase arg:" << caller->inputs()[i + 1]->DebugString() << ",index:" << i;
    }
  }
  TraceGuard trace_guard(std::make_shared<TraceCopy>(caller->debug_info()));
  auto new_caller = caller->func_graph()->NewCNode(new_args);
  new_caller->set_abstract(caller->abstract());
  // Should be done before manager. Replace as caller CNode will be dropped after Replace, the ReplaceInOrder will be
  // no effect.
  caller->func_graph()->ReplaceInOrder(caller, new_caller);
  (void)manager->Replace(caller, new_caller);
}

// Adjust the caller(returned tuple)'s caller(getitem call)'s caller of func graph.
// Since the elements in returned tuple maybe eliminated,
// we should convert getitem(returned_tuple, x) into the eliminating argument itself.
static inline void AdjustGetItemCall(const CNodePtr &caller,
                                     const mindspore::HashMap<size_t, size_t> &only_return_parameter_indexes) {
  const FuncGraphManagerPtr &manager = caller->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (only_return_parameter_indexes.empty()) {
    return;
  }
  const auto &node_users = manager->node_users();
  const auto &iter = node_users.find(caller);
  if (iter == node_users.end() || iter->second.empty()) {
    return;
  }
  std::vector<std::pair<AnfNodePtr, AnfNodePtr>> replacing_nodes;
  auto &all_users = iter->second;
  for (auto &user : all_users) {
    auto node = user.first;
    MS_EXCEPTION_IF_NULL(node);
    if (!IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      MS_LOG(ERROR) << "We expect a GetItem from the return tuple, but got " << node->DebugString();
      continue;
    }
    auto getitem_cnode = dyn_cast<CNode>(node);
    MS_EXCEPTION_IF_NULL(getitem_cnode);
    // Check if it's the eliminated element of returned tuple.
    constexpr size_t getitem_index_pos = 2;
    auto &index_node = getitem_cnode->input(getitem_index_pos);
    auto index_value = GetValueNode<Int64ImmPtr>(index_node);
    if (index_value == nullptr || index_value->value() < 0) {
      MS_LOG(EXCEPTION) << "The index_value is incorrect, " << index_node->DebugString();
    }
    size_t index_value_imm = LongToSize(index_value->value());
    const auto &index_pos = only_return_parameter_indexes.find(index_value_imm + 1);
    if (index_pos == only_return_parameter_indexes.end()) {
      continue;
    }

    // Found the tuple element, to replace it.
    auto eliminating_argument_pos = index_pos->second;
    MS_LOG(DEBUG) << "Found unused getitem CNode: " << getitem_cnode->DebugString() << ", index: " << index_value_imm
                  << ", eliminating_argument_pos: " << eliminating_argument_pos;
    // Replace the getitem CNode with the eliminated argument.
    auto &arg = caller->input(eliminating_argument_pos + 1);
    (void)replacing_nodes.emplace_back(std::pair(getitem_cnode, arg));
  }
  for (auto &nodes : replacing_nodes) {
    MS_LOG(DEBUG) << "Replace: " << nodes.first->DebugString() << ", with: " << nodes.second->DebugString();
    (void)manager->Replace(nodes.first, nodes.second);
  }
}

class ParameterEliminator {
 public:
  ParameterEliminator() = default;
  virtual ~ParameterEliminator() = default;
  bool operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &) {
    bool changes = false;
    while (true) {
      const auto &[fg, callers] = SearchFuncGraphCallers(func_graph, eliminate_only_returned_parameter_);
      if (fg == nullptr) {
        break;
      }
      const auto &[unused_parameter_indexes, only_return_parameter_indexes] =
        EraseUnusedParameters(fg, eliminate_only_returned_parameter_);
      for (auto caller : callers) {
        // Replace the getitem CNodes with the arguments.
        if (eliminate_only_returned_parameter_) {
          AdjustGetItemCall(caller, only_return_parameter_indexes);
        }
        // Erase the arguments for eliminated parameters.
        AdjustCallerArgs(caller, unused_parameter_indexes);
      }
      changes = true;
    }
    return changes;
  }

  void set_eliminate_only_returned_parameter(bool eliminate_only_returned_parameter) {
    eliminate_only_returned_parameter_ = eliminate_only_returned_parameter;
  }

 private:
  bool eliminate_only_returned_parameter_{false};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_PARAMETER_ELIMINATE_H
