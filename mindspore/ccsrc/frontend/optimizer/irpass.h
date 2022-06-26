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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_H_

#include <memory>

#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/opt.h"
#include "frontend/optimizer/anf_visitor.h"

namespace mindspore {
namespace opt {
namespace irpass {
// the collection of irpass for optimie action
class OptimizeIRPassLib {
 public:
  OptimizeIRPassLib();
  ~OptimizeIRPassLib() = default;

  SubstitutionPtr arithmetic_simplify_;
  SubstitutionPtr arithmetic_simplify2_;
  SubstitutionPtr special_op_eliminate_;
  SubstitutionPtr ad_related_special_op_eliminate_;
  SubstitutionPtr zero_like_fill_zero_;
  SubstitutionPtr adjust_all_reduce_mul_add_;
  SubstitutionPtr float_depend_g_call_;
  //  ops eliminate
  SubstitutionPtr tuple_list_get_item_eliminator_;
  SubstitutionPtr tuple_list_get_item_const_eliminator_;
  SubstitutionPtr tuple_list_set_item_eliminator_;
  SubstitutionPtr tuple_list_get_set_item_eliminator_;
  SubstitutionPtr tuple_list_get_item_depend_reorder_;
  SubstitutionPtr tuple_list_convert_item_index_to_positive_;
  SubstitutionPtr make_slice_get_slice_eliminator_;

  SubstitutionPtr stack_unstack_eliminate_;
  SubstitutionPtr tile_eliminate_;
  SubstitutionPtr cast_eliminate_;
  SubstitutionPtr reshape_eliminate_;
  SubstitutionPtr transpose_eliminate_;
  SubstitutionPtr reduce_eliminate_;
  SubstitutionPtr partial_eliminate_;
  SubstitutionPtr same_eliminate_;
  SubstitutionPtr check_bprop_eliminate_;
  SubstitutionPtr reset_defer_inline_;
  SubstitutionPtr depend_value_elim_;
  SubstitutionPtr all_reduce_const_elim_;
  SubstitutionPtr mirror_mini_step_elim_;
  SubstitutionPtr virtual_add_elim_;
  SubstitutionPtr mini_step_allgather_replace_;
  SubstitutionPtr micro_step_allgather_replace_;
  SubstitutionPtr real_op_eliminate_;

  // Env Item Eliminate
  SubstitutionPtr environ_get_eliminate_;
  SubstitutionPtr environ_get_add_eliminate_;
  SubstitutionPtr environ_get_set_eliminate_;
  SubstitutionPtr environ_get_depend_swap_;
  SubstitutionPtr environ_add_const_eliminate_;
  SubstitutionPtr incorporate_environ_get_;
  SubstitutionPtr incorporate_environ_get_bypass_recursive_;
  SubstitutionPtr incorporate_environ_get_switch_;
  SubstitutionPtr incorporate_environ_get_switch_layer_;
  SubstitutionPtr split_environ_get_set_with_tuple_value_;

  // Ref eliminate
  SubstitutionPtr make_ref_eliminate_;
  SubstitutionPtr get_ref_param_eliminate_;
  SubstitutionPtr get_make_ref_eliminate_;
  SubstitutionPtr replace_refkey_by_param_;
  SubstitutionPtr replace_old_param_;

  // Branch culling
  SubstitutionPtr switch_simplify_;
  SubstitutionPtr float_tuple_getitem_switch_;
  SubstitutionPtr float_environ_get_switch_;
  SubstitutionPtr exchange_switch_depend_value_;

  SubstitutionPtr switch_partial_eliminater_;
  SubstitutionPtr switch_layer_partial_eliminater_;

  // AddN
  SubstitutionPtr merge_addn_;
  SubstitutionPtr addn_zero_filter_;
  SubstitutionPtr addn_check_dump_;

  // AccumulateNV2
  SubstitutionPtr accumulaten_eliminater_;

  // Accelerated Algorithm
  SubstitutionPtr less_batch_normalization_;

  // Gradient irpasses
  SubstitutionPtr minmaximum_grad_;

  // inline
  SubstitutionPtr inline_;
  SubstitutionPtr inline_without_move_;
  SubstitutionPtr replace_applicator_;
  SubstitutionPtr specialize_transform_;

  // Auto-monad related eliminaters.
  SubstitutionPtr updatestate_useless_node_eliminater_;
  SubstitutionPtr updatestate_pure_node_eliminater_;
  SubstitutionPtr switch_call_monad_eliminater_;
  SubstitutionPtr stopgrad_eliminater_;
  SubstitutionPtr load_eliminater_;

  // Incorporation
  SubstitutionPtr incorporate_getitem_set_;
  SubstitutionPtr incorporate_call_;
  SubstitutionPtr incorporate_call_switch_;

  // virtual dataset
  SubstitutionPtr virtual_dataset_eliminate_;

  // virtual output
  SubstitutionPtr virtual_output_eliminate_;

  // PipelineSplit
  SubstitutionPtr receive_eliminate_;
  SubstitutionPtr virtual_accu_grad_;
  SubstitutionPtr virtual_assign_add_;
  SubstitutionPtr mirror_micro_step_;

  // Convert
  SubstitutionPtr print_tuple_wrapper_;

  // tuple parameter graph transform
  SubstitutionPtr call_graph_tuple_transform_;

  // RowTensor Eliminate
  SubstitutionPtr row_tensor_eliminate_;

  // RowTensorAddZerosLike Eliminate
  SubstitutionPtr row_tensor_add_zeros_like_;

  // SparseTensor Eliminate
  SubstitutionPtr sparse_tensor_eliminate_;

  // Value_Based Eliminate
  SubstitutionPtr value_based_eliminate_;

  // Switch defer inline
  SubstitutionPtr switch_defer_inline_;

  // SwitchLayer defer inline
  SubstitutionPtr switch_layer_defer_inline_;

  // Pynative Eliminate
  SubstitutionPtr pynative_eliminate_;

  // Recompute
  SubstitutionPtr set_cell_output_no_recompute_;

  // Workaround
  SubstitutionPtr stop_gradient_special_op_;

  // ge TensorArray process
  SubstitutionPtr ge_tensor_array_add_flow_;
  SubstitutionPtr ge_tensor_array_cast_index_;
};

// the collection of irpass for resolve action
class ResolveIRPassLib {
 public:
  ResolveIRPassLib();
  ~ResolveIRPassLib() = default;
  SubstitutionPtr resolver_;
};

class InferenceOptPrepareLib {
 public:
  InferenceOptPrepareLib();
  ~InferenceOptPrepareLib() = default;
  SubstitutionPtr grad_var_prepare_;
};

// predicate functions
inline bool IsNode(const AnfNodePtr &) { return true; }

inline bool IsCNode(const AnfNodePtr &node) {
  if (node != nullptr) {
    return node->isa<CNode>();
  }
  return false;
}

inline bool IsVNode(const AnfNodePtr &node) {
  if (node != nullptr) {
    return node->isa<ValueNode>();
  }
  return false;
}

inline bool IsParam(const AnfNodePtr &node) {
  if (node != nullptr) {
    return node->isa<Parameter>();
  }
  return false;
}

inline bool IsLoad(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return false;
  }
  return IsPrimitiveCNode(node, prim::kPrimLoad);
}

// Check if CNode Input 0 is Func Graph
inline bool IsCNodeGraph(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return false;
  }

  auto inp0 = node->cast<CNodePtr>()->input(0);
  return IsValueNode<FuncGraph>(inp0);
}

// Check if CNode Input 0 is CNode
inline bool IsCNodeDup(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return false;
  }

  auto inp0 = node->cast<CNodePtr>()->input(0);
  return (inp0 != nullptr) && inp0->isa<CNode>();
}

// check if the cnode is a switch cnode
inline bool IsCNodeSwitch(const AnfNodePtr &node) {
  if (node != nullptr) {
    if (node->isa<CNode>()) {
      return IsPrimitiveCNode(node, prim::kPrimSwitch);
    }
  }
  return false;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_H_
