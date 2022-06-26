/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INLINE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INLINE_H_

#include <vector>
#include <utility>
#include <algorithm>

#include "utils/hash_map.h"
#include "frontend/optimizer/irpass.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"
#include "ir/tensor.h"
#include "frontend/operator/ops.h"
#include "abstract/abstract_value.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
namespace irpass {
class ReplaceApplicator : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!IsValueNode<FuncGraph>(node)) {
      return nullptr;
    }

    auto fg = GetValueNode<FuncGraphPtr>(node);
    if (fg->has_flag(FUNC_GRAPH_FLAG_NO_INLINE) || fg->has_flag(FUNC_GRAPH_FLAG_DEFER_INLINE) || fg->stub() ||
        *(fg->switch_input()) || *(fg->switch_layer_input())) {
      return nullptr;
    }
    // Defer inlining in the case of pipeline.
    auto stage_num = parallel::ParallelContext::GetInstance()->pipeline_stage_split_num();
    if (fg->stage() != -1 && stage_num > 1) {
      return nullptr;
    }
    // Defer inlining to get the output nodes of the recomputed cell whose output is non-recomputed.
    if (fg->has_flag(FUNC_GRAPH_OUTPUT_NO_RECOMPUTE)) {
      return nullptr;
    }

    auto out = fg->output();
    MS_EXCEPTION_IF_NULL(out);
    if (!out->isa<CNode>()) {
      return nullptr;
    }

    auto &inputs = out->cast<CNodePtr>()->inputs();
    auto params = fg->parameters();

    // Exclude first elements of inputs which is fn.
    auto input_size = inputs.size();
    auto param_size = params.size();
    if ((input_size == 1 && param_size == 0) || (input_size > 1 && (input_size - 1) == param_size &&
                                                 std::equal(inputs.begin() + 1, inputs.end(), params.begin()))) {
      auto inner = inputs[0];
      if (IsValueNode<Primitive>(inner) ||
          (IsValueNode<FuncGraph>(inner) && GetValueNode<FuncGraphPtr>(inner)->parent() == nullptr)) {
        return inner;
      }
    }

    return nullptr;
  }
};

class InlinerBase;
using CriterionFuncType = std::function<bool(InlinerBase *, const FuncGraphPtr &, const AnfNodePtr &)>;

bool IsUniqueUse(InlinerBase *, const FuncGraphPtr &fg, const AnfNodePtr &);

bool IsTrivial(InlinerBase *, const FuncGraphPtr &fg, const AnfNodePtr &);
bool IsInside(InlinerBase *, const FuncGraphPtr &, const AnfNodePtr &node);
bool IsCore(InlinerBase *, const FuncGraphPtr &fg, const AnfNodePtr &);
bool IsDirectParentCall(InlinerBase *, const FuncGraphPtr &fg, const AnfNodePtr &node);
bool IsNotRecursive(InlinerBase *inliner, const FuncGraphPtr &fg, const AnfNodePtr &);
bool IsForceInline(InlinerBase *, const FuncGraphPtr &fg, const AnfNodePtr &) {
  return fg->has_flag(FUNC_GRAPH_FLAG_FORCE_INLINE);
}

// {G, Xs}
class InlinerBase : public AnfVisitor {
 public:
  explicit InlinerBase(const std::vector<std::vector<CriterionFuncType>> &criterions, bool use_move = true)
      : use_move_(use_move), criterions_(criterions) {}
  ~InlinerBase() override = default;
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    auto cnode = dyn_cast<CNode>(node);
    if (cnode == nullptr || cnode->size() < 1) {
      return nullptr;
    }

    auto &inputs = cnode->inputs();
    // G
    auto fg = GetValueNode<FuncGraphPtr>(inputs[0]);
    if (fg == nullptr || fg->has_flag(FUNC_GRAPH_FLAG_NO_INLINE) || fg->has_flag(FUNC_GRAPH_FLAG_DEFER_INLINE) ||
        fg->stub()) {
      return nullptr;
    }
    // Defer inlining in the case of pipeline.
    auto stage_num = parallel::ParallelContext::GetInstance()->pipeline_stage_split_num();
    if (fg->stage() != -1 && stage_num > 1) {
      return nullptr;
    }
    // Defer inlining to get the output nodes of the recomputed cell whose output is non-recomputed.
    if (fg->has_flag(FUNC_GRAPH_OUTPUT_NO_RECOMPUTE)) {
      return nullptr;
    }

    Reset();

    // 'criterions_': {criterion_group_1:{criterion1, criterion2, ...}, criterion_group_2:{...}, ...}
    // All the criterions of 'criterion group' are true would set 'criterion group' as 'true'. As [AND].
    // Anyone of 'criterion group' in 'criterions_' is 'true' would be matched. As [OR].
    bool is_match = ApplyCriterions(node, fg);
    if (!is_match) {
      return nullptr;
    }

    std::vector<AnfNodePtr> args;
    (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(args));
    // Compare size to avoid the case that the function has default value after grad.
    // for which after renormalize, the function default value will be an input
    if (fg->parameters().size() != args.size()) {
      return nullptr;
    }

    if (IsForceInline(this, fg, node)) {
      if (IsUniqueUse(nullptr, fg, nullptr)) {
        return InlineMove(node, fg, args, inputs);
      }
      return InlineClone(fg, node->func_graph(), args, inputs[0]->scope());
    }

    if (IsUniqueUse(nullptr, fg, nullptr)) {
      // For the single used fg, including non-after and after not matched above,
      // we move the whole fg nodes.
      auto ret_node = InlineForUniqueUse(node, fg, args, inputs);
      if (ret_node != nullptr) {
        return ret_node;
      }
    } else {
      // We don't expand the middle multiple used after block, except the last one.
      if (GraphHasBranch(fg)) {
        return nullptr;
      }
      // Check if parameters' changed for the first met branch calling.
      if (fg->has_flag(FUNC_GRAPH_FLAG_AFTER_BLOCK)) {
        auto param_simplified_caller = SimplifyAfterParameter(fg, node, args);
        if (param_simplified_caller != nullptr) {
          return param_simplified_caller;
        }
      }
    }
    // Or, just make a clone for not single used fg.
    MS_LOG(DEBUG) << "Run InlineClone in inline pass, subgraph number may increase.";
    return InlineClone(fg, node->func_graph(), args, inputs[0]->scope());
  }

  AnfNodePtr InlineMove(const AnfNodePtr &node, const FuncGraphPtr &fg, const std::vector<AnfNodePtr> &args,
                        const std::vector<AnfNodePtr> &inputs) {
    auto mng = fg->manager();
    MS_EXCEPTION_IF_NULL(mng);
    ReplaceParams(mng, args, fg);
    auto out_node = fg->output();
    mng->MoveAllCNodeDropGraph(fg, node->func_graph(), inputs[0]->scope());
    return out_node;
  }

  AnfNodePtr InlineForUniqueUse(const AnfNodePtr &node, const FuncGraphPtr &fg, const std::vector<AnfNodePtr> &args,
                                const std::vector<AnfNodePtr> &inputs) {
    if (use_move_) {
      return InlineMove(node, fg, args, inputs);
    }

    // The other branch calling the last after block.
    if (fg->has_flag(FUNC_GRAPH_FLAG_AFTER_BLOCK)) {
      // Check if parameters' changed.
      auto param_simplified_caller = SimplifyAfterParameter(fg, node, args);
      if (param_simplified_caller != nullptr) {
        return param_simplified_caller;
      }
    }
    return nullptr;
  }

  bool ApplyCriterions(const AnfNodePtr &node, const FuncGraphPtr &fg) {
    bool is_match = false;
    for (auto &criterions : criterions_) {  // Each 'criterion group' in criterions_.
      is_match = true;
      for (auto &criterion : criterions) {  // Each criterion in 'criterion group'.
        if (!criterion(this, fg, node)) {
          is_match = false;
          break;
        }
      }
      if (is_match) {
        break;
      }
    }
    return is_match;
  }

  void ReplaceParams(const FuncGraphManagerPtr &mng, const std::vector<AnfNodePtr> &new_params,
                     const FuncGraphPtr &fg) {
    auto params = fg->parameters();
    auto old_size = params.size();
    if (old_size != new_params.size()) {
      MS_LOG(EXCEPTION) << "Parameter size not match." << old_size << " new " << new_params.size()
                        << fg->output()->DebugString(10);
    }
    for (size_t i = 0; i < old_size; i++) {
      (void)mng->Replace(params[i], new_params[i]);
    }
  }

  bool IsRecursive(const FuncGraphPtr &fg) {
    if (!is_checked_) {
      is_checked_ = true;
      is_recursive_ = fg->recursive();
    }
    return is_recursive_;
  }

  void Reset() {
    is_checked_ = false;
    is_recursive_ = false;
  }

  // For after block which contains branch call, delete the parameters which is not used.
  // In most cases, it may be a `Module` or other constant input.
  AnfNodePtr SimplifyAfterParameter(const FuncGraphPtr &fg, const AnfNodePtr &node,
                                    const std::vector<AnfNodePtr> &args) {
    auto &fg_params = fg->parameters();
    std::vector<int64_t> used_param_index;
    auto mng = fg->manager();
    MS_EXCEPTION_IF_NULL(mng);
    bool should_simplify = false;
    for (size_t i = 0; i < fg_params.size(); i++) {
      if (mng->node_users()[fg_params[i]].size() != 0) {
        used_param_index.emplace_back(i);
      } else {
        MS_LOG(DEBUG) << "Not used parameter " << fg_params[i]->DebugString() << " for calling " << fg->ToString();
        should_simplify = true;
      }
    }
    if (!should_simplify) {
      return nullptr;
    }
    MS_LOG(DEBUG) << "Parameter not used found for graph :" << fg->ToString();
    // Clone a new graph and ignore the not used parameters
    auto new_fg = TransformableClone(fg);
    auto &new_fg_params = new_fg->parameters();
    std::vector<AnfNodePtr> new_params;
    std::transform(used_param_index.begin(), used_param_index.end(), std::back_inserter(new_params),
                   [&new_fg_params](size_t i) { return new_fg_params[i]; });
    new_fg->set_parameters(new_params);

    std::vector<AnfNodePtr> node_inputs;
    node_inputs.push_back(NewValueNode(new_fg));
    std::transform(used_param_index.begin(), used_param_index.end(), std::back_inserter(node_inputs),
                   [&args](size_t i) { return args[i]; });
    return node->func_graph()->NewCNode(node_inputs);
  }

  bool CheckSwitchBranchAbstract(const AbstractBasePtr &branch_abstract) {
    if (branch_abstract != nullptr && branch_abstract->isa<abstract::AbstractError>()) {
      auto branch_abstract_value = branch_abstract->GetValueTrack();
      MS_EXCEPTION_IF_NULL(branch_abstract_value);
      auto branch_abstract_value_string_imm = branch_abstract_value->cast<StringImmPtr>();
      if (branch_abstract_value_string_imm != nullptr) {
        auto branch_abstract_value_string_imm_value = branch_abstract_value_string_imm->value();
        return branch_abstract_value_string_imm_value == kDeadNodeName ||
               branch_abstract_value_string_imm_value == kPolyNodeName;
      }
    }
    return false;
  }

  bool CheckSwitchInputs(const std::vector<AnfNodePtr> &sw_inputs) {
    auto true_branch_abstract = sw_inputs[kSwitchTrueBranchIndex]->abstract();
    auto false_branch_abstract = sw_inputs[kSwitchFalseBranchIndex]->abstract();
    // When branch has dead node or poly node, do not perform inline.
    if (CheckSwitchBranchAbstract(true_branch_abstract) || CheckSwitchBranchAbstract(false_branch_abstract)) {
      return true;
    }
    return !sw_inputs[1]->isa<ValueNode>() || IsValueNode<tensor::Tensor>(sw_inputs[1]);
  }

  // This is a try-best algorithm to find a graph which may generate branch call.
  // It does not handle high-order function call. For high-orderer call branch, it still may be inlined.
  bool GraphHasBranch(const FuncGraphPtr &fg) {
    if (graph_branch_cache_.find(fg) != graph_branch_cache_.end()) {
      return graph_branch_cache_[fg];
    }
    bool has_branch = false;
    auto nodes = fg->nodes();
    for (auto &item : nodes) {
      if (IsPrimitiveCNode(item, prim::kPrimSwitch)) {
        auto sw_inputs = item->cast<CNodePtr>()->inputs();
        if (sw_inputs.size() != 4) {
          MS_LOG(EXCEPTION) << "switch inputs should be 4";
        }
        if (CheckSwitchInputs(sw_inputs)) {
          has_branch = true;
          break;
        }
      } else if (IsCNodeGraph(item)) {
        auto cinputs = item->cast<CNodePtr>()->inputs();
        if (cinputs.size() < 1) {
          MS_LOG(EXCEPTION) << "graph call inputs should be greater than 1";
        }
        FuncGraphPtr call_fg = GetValueNode<FuncGraphPtr>(cinputs[0]);
        bool call_fg_has_branch = GraphHasBranch(call_fg);
        if (call_fg_has_branch) {
          has_branch = true;
          break;
        }
      } else if (IsPrimitiveCNode(item, prim::kPrimPartial)) {
        auto cinputs = item->cast<CNodePtr>()->inputs();
        if (cinputs.size() < 2) {
          MS_LOG(EXCEPTION) << "partial call inputs should be greater than 2";
        }
        FuncGraphPtr call_fg = GetValueNode<FuncGraphPtr>(cinputs[1]);
        if (call_fg == nullptr) {
          continue;
        }
        bool call_fg_has_branch = GraphHasBranch(call_fg);
        if (call_fg_has_branch) {
          has_branch = true;
          break;
        }
      }
    }
    graph_branch_cache_[fg] = has_branch;
    return has_branch;
  }

 private:
  bool is_checked_{false}, is_recursive_{false};
  bool use_move_;
  std::vector<std::vector<CriterionFuncType>> criterions_;
  mindspore::HashMap<FuncGraphPtr, bool> graph_branch_cache_;
};

bool IsUniqueUse(InlinerBase *, const FuncGraphPtr &fg, const AnfNodePtr &) {
  const auto &users = fg->func_graph_cnodes_index();
  int64_t n_use = std::accumulate(
    users.begin(), users.end(), 0,
    [](int64_t sum, const std::pair<const CNodeIndexPairPtr, int64_t> &item) { return sum + item.second; });
  return n_use == 1;
}

bool IsTrivial(InlinerBase *, const FuncGraphPtr &fg, const AnfNodePtr &) {
  auto n_cnode = fg->nodes().size() - fg->parameters().size();
  // There is at least one CNode(return, other_node).
  return n_cnode <= 2;
}

bool IsInside(InlinerBase *, const FuncGraphPtr &, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node->func_graph());
  return node->func_graph()->has_flag("inline_inside");
}

bool IsCore(InlinerBase *, const FuncGraphPtr &fg, const AnfNodePtr &) { return fg->has_flag("core"); }

bool IsDirectParentCall(InlinerBase *, const FuncGraphPtr &fg, const AnfNodePtr &node) {
  bool unique_use = IsUniqueUse(nullptr, fg, nullptr);
  bool is_recursive = fg->recursive();
  if (fg->parent() != nullptr && is_recursive) {
    if (fg->parent() == node->func_graph() && unique_use) {
      return true;
    }
  }
  return false;
}

bool IsNotRecursive(InlinerBase *inliner, const FuncGraphPtr &fg, const AnfNodePtr &) {
  return !inliner->IsRecursive(fg);
}

class Inliner : public InlinerBase {
 public:
  explicit Inliner(bool use_move = true)
      : InlinerBase(
          // Supports AND conditions in one criterion, Ex. {IsUniqueUse, IsNotRecursive}.
          {
            {IsTrivial},
            {IsInside},
            {IsCore},
            {IsNotRecursive},
            {IsDirectParentCall},
          },
          use_move) {}

  ~Inliner() override = default;
};

class DirectInliner : public InlinerBase {
 public:
  explicit DirectInliner(bool use_move = true)
      : InlinerBase(
          // Supports AND conditions in one criterion, Ex. {IsUniqueUse, IsNotRecursive}.
          {
            {IsForceInline},
            {IsDirectParentCall},
          },
          use_move) {}
  ~DirectInliner() override = default;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INLINE_H_
