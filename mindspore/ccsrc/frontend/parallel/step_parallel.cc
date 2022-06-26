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

#include "frontend/parallel/step_parallel.h"

#include <inttypes.h>
#include <sys/time.h>
#include <algorithm>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <queue>

#include "utils/hash_map.h"
#include "base/core_ops.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/graph_util/pipeline_split_utils.h"
#include "frontend/parallel/node_check.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/ops_info/matmul_info.h"
#include "ir/param_info.h"
#include "ir/tensor.h"
#include "utils/trace_base.h"
#include "include/common/utils/comm_manager.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"
#include "mindspore/core/utils/parallel_node_check.h"
#include "frontend/parallel/parallel_optimizer/opt_param_mgr.h"
#if ((defined ENABLE_CPU) && (!defined _WIN32))
#include "ps/util.h"
#include "ps/ps_context.h"
#endif

using mindspore::tensor::Tensor;

namespace mindspore {
namespace parallel {
static const std::set<std::string> COMMUNICATION_OPS = {ALL_REDUCE, ALL_GATHER, ALL_TO_ALL, REDUCE_SCATTER};
static const std::set<std::string> INVALID_LOSS_OPS = {GET_NEXT, VIRTUALLOSS, LOAD, UPDATESTATE};
static const std::set<std::string> NO_INPUT_TENSOR_OPS = {UNIFORM_REAL};
// g_RefMap, for CNode B input i is a RefKey[Parameter C],
// it will be one item in map with key: C, and value: (B, i)
std::map<AnfNodePtr, std::pair<AnfNodePtr, int64_t>> g_RefMap;
const uint32_t MAX_BFS_DEPTH = 7;

void SetMiniStepOpDoMirrorLabel(std::vector<AnfNodePtr> new_node_input, bool do_mirror, bool accu_flag) {
  if (new_node_input.empty()) {
    return;
  }
  auto prim_anf_node = new_node_input[0]->cast<ValueNodePtr>();
  auto prim = GetValueNode<PrimitivePtr>(prim_anf_node);
  MS_EXCEPTION_IF_NULL(prim);

  auto attrs = prim->attrs();
  attrs[DO_MIRROR] = MakeValue<bool>(do_mirror);
  attrs[ADD_ACCU] = MakeValue<bool>(accu_flag);
  prim->SetAttrs(attrs);
}

void SetAllReduceRecomputeFlag(const std::vector<AnfNodePtr> &new_node_input, const CNodePtr &node) {
  if (new_node_input.empty()) {
    return;
  }

  auto prim_anf_node = new_node_input[0]->cast<ValueNodePtr>();
  auto prim = GetValueNode<PrimitivePtr>(prim_anf_node);
  MS_EXCEPTION_IF_NULL(prim);
  auto attrs = prim->attrs();

  auto anf_node = node->input(0)->cast<ValueNodePtr>();
  auto prim_node = GetValueNode<PrimitivePtr>(anf_node);
  MS_EXCEPTION_IF_NULL(prim_node);
  auto node_attrs = prim_node->attrs();
  if (node_attrs.find(RECOMPUTE_COMM_OP) != node_attrs.end() && !GetValue<bool>(node_attrs[RECOMPUTE_COMM_OP])) {
    attrs[RECOMPUTE] = MakeValue<bool>(false);
    prim->SetAttrs(attrs);
    MS_LOG(INFO) << "Do not recompute the forward communication operator of " << prim_node->ToString();
  }
}

std::vector<AnfNodePtr> CreateInput(const Operator &op, const AnfNodePtr &node, const std::string &instance_name) {
  MS_EXCEPTION_IF_NULL(node);
  OperatorArgs arg_forward = op.second;
  ValuePtr pyop_instance = CreateOpInstance(arg_forward.first, op.first, instance_name);
  MS_EXCEPTION_IF_NULL(pyop_instance);
  OperatorParams params = arg_forward.second;

  std::vector<AnfNodePtr> new_node_input = {NewValueNode(pyop_instance), node};
  if (!params.empty()) {
    for (auto &param : params) {
      AnfNodePtr val = NewValueNode(param.first.second);
      MS_EXCEPTION_IF_NULL(val);
      int64_t position = param.second;
      (void)new_node_input.insert(new_node_input.begin() + position, val);
    }
  }

  // if the op have 'group' attr, set the rank list name for the op
  SetCommunicationOpGroupLabel(new_node_input);
  return new_node_input;
}

AnfNodePtr GetAccuGrad(const std::vector<AnfNodePtr> &parameters, const std::string &weight_name) {
  for (auto &param : parameters) {
    if (!ParameterIsCloned(param)) {
      continue;
    }

    auto param_ptr = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    if (param_ptr->name().find(weight_name) != std::string::npos &&
        param_ptr->name().find(ACCU_GRADS) != std::string::npos) {
      MS_LOG(INFO) << "Find the accumulation grad node: " << param_ptr->name();
      return param;
    }
  }
  return nullptr;
}

std::vector<AnfNodePtr> CreateMirrorInput(const FuncGraphPtr &root, const Operator &op, const AnfNodePtr &node,
                                          const std::string &instance_name, const std::string &weight_name) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(root->manager());

  std::string op_name = op.first;
  OperatorArgs arg_forward = op.second;
  AnfNodePtr grad_accu = nullptr;

  int64_t grad_accumulation_step = ParallelContext::GetInstance()->grad_accumulation_step();
  int64_t split_stage_num = ParallelContext::GetInstance()->pipeline_stage_split_num();

  if (grad_accumulation_step > 1 || split_stage_num > 1) {
    auto parameters = root->parameters();
    grad_accu = GetAccuGrad(parameters, weight_name);
    if (!grad_accu) {
      if (op_name == MIRROR_MINI_STEP_OPERATOR) {
        op_name = MIRROR_OPERATOR;
        arg_forward.first.pop_back();
      } else if (op_name == MINI_STEP_ALL_GATHER || op_name == MIRROR_MICRO_STEP_OPERATOR ||
                 op_name == MICRO_STEP_ALL_GATHER) {
        MS_LOG(EXCEPTION) << "You should define `accu_grads` when use " << op_name << " parameter:" << weight_name;
      }
    }
  }

  ValuePtr pyop_instance = CreateOpInstance(arg_forward.first, op_name, instance_name);
  MS_EXCEPTION_IF_NULL(pyop_instance);
  OperatorParams params = arg_forward.second;

  std::vector<AnfNodePtr> new_node_input;
  if (op_name == MIRROR_MINI_STEP_OPERATOR || op_name == MINI_STEP_ALL_GATHER ||
      op_name == MIRROR_MICRO_STEP_OPERATOR || op_name == MICRO_STEP_ALL_GATHER) {
    new_node_input = {NewValueNode(pyop_instance), node, grad_accu};
    MS_LOG(INFO) << "Insert the grad accumulation node as the mirror op's input";
  } else {
    new_node_input = {NewValueNode(pyop_instance), node};
  }

  if (!params.empty()) {
    for (auto &param : params) {
      AnfNodePtr val = NewValueNode(param.first.second);
      MS_EXCEPTION_IF_NULL(val);
      int64_t position = param.second;
      (void)new_node_input.insert(new_node_input.begin() + position, val);
    }
  }

  // if the op have 'group' attr, set the rank list name for the op
  SetCommunicationOpGroupLabel(new_node_input);
  // gradient accumulation
  if (grad_accumulation_step > 1) {
    bool add_accu = root->has_flag(kAccumulation);
    // MiniStep need to do mirror at each micro step as we use the gradient accumulation sharding,
    SetMiniStepOpDoMirrorLabel(new_node_input, !add_accu, !add_accu);
  }
  return new_node_input;
}

void InsertNode(const Operator &op, const CNodePtr &node, size_t index, const AnfNodePtr &pre_node,
                const FuncGraphPtr &func_graph, const std::string &instance_name, const std::string &param_name = "",
                const FuncGraphPtr &root = nullptr) {
  // insert new node before the node
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  ScopePtr scope = node->scope();
  MS_EXCEPTION_IF_NULL(scope);
  std::vector<AnfNodePtr> node_input;
  if (root && !param_name.empty()) {
    node_input = CreateMirrorInput(root, op, pre_node, instance_name, param_name);
  } else {
    node_input = CreateInput(op, pre_node, instance_name);
  }
  CNodePtr new_node = func_graph->NewCNode(node_input);
  MS_EXCEPTION_IF_NULL(new_node);
  if (instance_name.find(SPLIT_SENS) == std::string::npos) {
    new_node->set_in_forward_flag(true);  // mark forward flag
  }
  auto new_node_value = node_input[0]->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(new_node_value);
  PrimitivePtr new_node_prim = new_node_value->value()->cast<PrimitivePtr>();
  new_node_prim->set_instance_name(instance_name);
  new_node_prim->set_attr("keep_value_node_input", MakeValue(true));
  if (instance_name.find(NOT_RECOMPUTE) != std::string::npos) {
    new_node_prim->set_attr("recompute", MakeValue(false));
  }
  new_node->set_scope(scope);
  node_input[0]->set_scope(scope);
  manager->SetEdge(node, SizeToInt(index), new_node);
  MS_LOG(INFO) << "Insert " << instance_name << " success";
}

// Replace pre_node with pre_node->op
static CNodePtr ReplaceNode(const Operator &op, const AnfNodePtr &pre_node, const FuncGraphPtr &func_graph,
                            const std::string &instance_name, const std::string &param_name = "",
                            const FuncGraphPtr &root = nullptr) {
  // insert new node before the node
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  ScopePtr scope = pre_node->scope();
  MS_EXCEPTION_IF_NULL(scope);
  std::vector<AnfNodePtr> node_input;
  if (root && !param_name.empty()) {
    node_input = CreateMirrorInput(root, op, pre_node, instance_name, param_name);
  } else {
    node_input = CreateInput(op, pre_node, instance_name);
  }
  CNodePtr new_node = func_graph->NewCNode(node_input);
  MS_EXCEPTION_IF_NULL(new_node);
  if (instance_name.find(SPLIT_SENS) == std::string::npos) {
    new_node->set_in_forward_flag(true);  // mark forward flag
  }
  auto new_node_prim = GetValueNode<PrimitivePtr>(node_input[0]);
  new_node_prim->set_instance_name(instance_name);
  new_node_prim->set_attr("keep_value_node_input", MakeValue(true));
  if (instance_name.find(NOT_RECOMPUTE) != std::string::npos) {
    new_node_prim->set_attr("recompute", MakeValue(false));
  }
  new_node->set_scope(scope);
  node_input[0]->set_scope(scope);
  manager->Replace(pre_node, new_node);
  MS_LOG(INFO) << "Insert " << instance_name << " success";
  return new_node;
}

void ForwardCommunication(OperatorVector forward_op, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  // step1:get graph manager distribute_operator
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto uses_set = manager->node_users()[node];
  CNodePtr node_to_insert = node;
  for (auto &uses_pair : uses_set) {
    auto uses_cnode = uses_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(uses_cnode);
    if (!IsValueNode<Primitive>(uses_cnode->input(0))) {
      break;
    }
    PrimitivePtr value_node_prim = GetValueNode<PrimitivePtr>(uses_cnode->input(0));
    MS_EXCEPTION_IF_NULL(value_node_prim);
    if (value_node_prim->name() == prim::kTupleGetItem) {
      if (uses_set.size() > 1) {
        MS_LOG(EXCEPTION) << "Now only support one output, but got " << uses_set.size();
      }
      node_to_insert = uses_cnode;
    }
  }
  MS_EXCEPTION_IF_NULL(node_to_insert);
  std::reverse(forward_op.begin(), forward_op.end());

  // step2:traverse op_list and insert node
  for (size_t index = 0; index < forward_op.size(); ++index) {
    std::string instance_name_base = FORWARD_OP;
    std::string instance_name = instance_name_base + "_" + CreateInstanceName(node, index);
    std::vector<AnfNodePtr> forward_input = CreateInput(forward_op[index], node_to_insert, instance_name);
    SetAllReduceRecomputeFlag(forward_input, node_to_insert);
    CNodePtr forward_node = func_graph->NewCNode(forward_input);  // using NewCNode to create anfnode
    MS_EXCEPTION_IF_NULL(forward_node);
    ScopePtr scope = node->scope();
    MS_EXCEPTION_IF_NULL(scope);
    forward_node->set_scope(scope);
    forward_node->set_in_forward_flag(true);
    forward_input[0]->set_scope(scope);
    (void)manager->Replace(node_to_insert, forward_node);  // using Replace function to insert node
  }
}

CNodePtr InsertMakeTuple(const AnfNodePtr &prev, uint64_t num, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(prev);
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> make_tuple_inputs;
  make_tuple_inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
  for (uint64_t i = 0; i < num; i++) {
    std::vector<AnfNodePtr> tuple_get_item_inputs{NewValueNode(prim::kPrimTupleGetItem), prev,
                                                  CreatInt64Imm(UlongToLong(i))};
    auto tuple_get_item = func_graph->NewCNode(tuple_get_item_inputs);
    MS_EXCEPTION_IF_NULL(tuple_get_item);
    make_tuple_inputs.push_back(tuple_get_item);
  }
  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(prev, make_tuple);
  return make_tuple;
}

void InsertRedistribution(const RedistributionOpListPtr &redistribution_oplist_ptr, const CNodePtr &node,
                          const FuncGraphPtr &func_graph, int64_t pos, const CNodePtr &pre_node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(pre_node);
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if ((redistribution_oplist_ptr->first).size() != (redistribution_oplist_ptr->second).size()) {
    MS_LOG(EXCEPTION) << "size of OperatorVector and OutPutInfoVector must be the same!";
  }
  for (size_t index = 0; index < (redistribution_oplist_ptr->first).size(); ++index) {
    if (pos >= SizeToLong(node->inputs().size())) {
      MS_LOG(EXCEPTION) << "InsertRedistribution:pos can't be larger than node's inputs'size";
    }
    // Create new node
    AnfNodePtr target_node = node->input(LongToSize(pos));
    MS_EXCEPTION_IF_NULL(target_node);
    // Create instance_name
    auto op = (redistribution_oplist_ptr->first)[index];
    std::string op_name = (redistribution_oplist_ptr->first)[index].first;
    std::string instance_name_base = REDISTRIBUTION_OP;
    std::string instance_name = instance_name_base + "_" + CreateInstanceName(pre_node, index) + op_name;
    auto prim_out = GetCNodePrimitive(node);
    auto prim_in = GetCNodePrimitive(pre_node);
    if (prim_out != nullptr && prim_in != nullptr) {
      auto prim_out_attr = prim_out->attrs();
      auto prim_in_attr = prim_in->attrs();
      if (((prim_out_attr.find(RECOMPUTE_COMM_OP) != prim_out_attr.end() &&
            !GetValue<bool>(prim_out_attr[RECOMPUTE_COMM_OP])) ||
           (prim_in_attr.find(RECOMPUTE_COMM_OP) != prim_in_attr.end() &&
            !GetValue<bool>(prim_in_attr[RECOMPUTE_COMM_OP]))) &&
          COMMUNICATION_OPS.find(op_name) != COMMUNICATION_OPS.end()) {
        MS_LOG(INFO) << "The redistribution node would not be recomputed.";
        instance_name = instance_name + "_" + NOT_RECOMPUTE;
      }
    }
    InsertNode(op, node, LongToSize(pos), target_node, func_graph, instance_name);
    if ((redistribution_oplist_ptr->second)[index].first) {
      target_node = node->input(LongToSize(pos));
      MS_EXCEPTION_IF_NULL(target_node);
      (void)InsertMakeTuple(target_node, (redistribution_oplist_ptr->second)[index].second, func_graph);
    }
  }
}

void InsertGetTensorSliceOp(const Operator &op, const CNodePtr &node, const FuncGraphPtr &func_graph, int64_t pos,
                            const std::string &instance_name) {
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "InsertGetTensorSliceOp: the graph is null, the instance name is " << instance_name;
  }

  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (pos >= SizeToLong(node->inputs().size())) {
    MS_LOG(EXCEPTION) << "InsertGetTensorSliceOp: pos can't be larger than node's inputs'size, the instance name is "
                      << instance_name;
  }
  // Create new node
  AnfNodePtr pre_node = node->input(LongToSize(pos));
  MS_EXCEPTION_IF_NULL(pre_node);
  InsertNode(op, node, LongToSize(pos), pre_node, func_graph, instance_name);
}

TensorLayout GetTensorInLayout(const CNodePtr &middle_node, const PrimitivePtr &middle_prim,
                               const OperatorInfoPtr &distribute_operator) {
  TensorInfo tensorinfo_in;
  if (middle_prim->name() == prim::kTupleGetItem) {
    auto value_node = middle_node->input(2)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    size_t index_s = LongToSize(GetValue<int64_t>(value_node->value()));
    if (index_s >= distribute_operator->outputs_tensor_info().size()) {
      MS_LOG(EXCEPTION) << "The index out of range, index: " << index_s
                        << ", vector size: " << distribute_operator->outputs_tensor_info().size();
    }
    tensorinfo_in = distribute_operator->outputs_tensor_info()[index_s];
  } else {
    if (distribute_operator->outputs_tensor_info().empty()) {
      MS_LOG(EXCEPTION) << "The outputs tensor info is empty";
    }
    tensorinfo_in = distribute_operator->outputs_tensor_info()[0];
  }
  return tensorinfo_in.tensor_layout();
}

OperatorInfoPtr GetDistributeOperator(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsParallelCareNode(node)) {
    return nullptr;
  }
  OperatorInfoPtr distribute_operator = node->user_data<OperatorInfo>();
  return distribute_operator;
}

void Redistribution(const std::pair<AnfNodePtr, int64_t> &node_pair, const OperatorInfoPtr &distribute_operator,
                    const CNodePtr &middle_node, int64_t index, TensorRedistribution tensor_redistribution,
                    const CNodePtr &pre_node) {
  FuncGraphPtr func_graph = middle_node->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Redistribution:get graph failed";
  }
  CNodePtr next_node = node_pair.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(next_node);
  auto middle_value = middle_node->input(0)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(middle_value);
  PrimitivePtr middle_prim = middle_value->value()->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(middle_prim);
  OperatorInfoPtr next_distribute_operator = GetDistributeOperator(next_node);
  if (next_distribute_operator == nullptr) {
    MS_LOG(EXCEPTION) << "Failure: " << next_node->ToString() << " GetDistributeOperator failed";
  }
  RankList dev_list = distribute_operator->stage_device_list();
  std::string next_prim_name = GetValueNode<PrimitivePtr>(next_node->input(0))->name();
  MS_LOG(DEBUG) << "Redistribution: middle_prim " << middle_prim->name() << " next_prim " << next_prim_name;
  MS_LOG(DEBUG) << "Redistribution: middle_node " << middle_node->ToString() << " next_node " << next_node->ToString();
  // extract tensor layout in and out
  if (distribute_operator->outputs_tensor_info().empty()) {
    MS_LOG(WARNING) << "pre_node's tensorinfo_in is empty, operator name is " << distribute_operator->name();
    return;
  }

  if (LongToSize(index - 1) >= next_distribute_operator->inputs_tensor_info().size()) {
    MS_LOG(WARNING) << "The index is out of range, the index is " << (index - 1) << ", the vector size is "
                    << next_distribute_operator->inputs_tensor_info().size() << "next operator name is "
                    << next_distribute_operator->name();
    return;
  }
  TensorInfo tensorinfo_out = next_distribute_operator->inputs_tensor_info()[LongToSize(index - 1)];
  TensorLayout tensorlayout_out = tensorinfo_out.tensor_layout();
  TensorLayout tensorlayout_in = GetTensorInLayout(middle_node, middle_prim, distribute_operator);
  if (IsPrimitiveCNode(middle_node, prim::kPrimReceive)) {
    tensorlayout_in = *(middle_node->user_data<TensorLayout>());
  }
  if (tensor_redistribution.Init(tensorlayout_in, tensorlayout_out, dev_list) == FAILED) {
    MS_LOG(ERROR) << "Redistribution: middle_prim " << middle_prim->name() << " next_prim : " << next_prim_name;
    MS_LOG(ERROR) << "Redistribution: middle_node " << middle_node->ToString() << " next_node "
                  << next_node->ToString();
    DumpGraph(func_graph, "redistribution_error");
    MS_LOG(EXCEPTION) << "Failure:tensor_redistribution init failed";
  }
  RedistributionOpListPtr redistribution_oplist_ptr = tensor_redistribution.InferTensorRedistributionOperatorList();
  if (redistribution_oplist_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:InferTensorRedistribution failed";
  }
  MS_LOG(DEBUG) << "Redistribution size " << redistribution_oplist_ptr->first.size();
  if (!redistribution_oplist_ptr->first.empty()) {
    // insert node before next node
    InsertRedistribution(redistribution_oplist_ptr, next_node, func_graph, node_pair.second, pre_node);
  }
}

bool StrategyFound(const mindspore::HashMap<std::string, ValuePtr> &attrs) {
  auto iter = attrs.find(IN_STRATEGY);
  return !((iter == attrs.end()) || (iter->second->type_name() == NONE));
}

bool AttrFound(const mindspore::HashMap<std::string, ValuePtr> &attrs, const std::string &target) {
  auto iter = attrs.find(target);
  return !((iter == attrs.end()) || (iter->second->type_name() == NONE));
}

bool HasStrategy(const FuncGraphPtr &root) {
  AnfNodePtr ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);

  for (auto &node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if ((cnode == nullptr) || !IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }

    ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);
    auto attrs = prim->attrs();
    if (StrategyFound(attrs)) {
      return true;
    }
  }

  return false;
}

bool IsCommunicationOp(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  return (COMMUNICATION_OPS.find(prim->name()) != COMMUNICATION_OPS.end());
}

bool FindCommunicationOp(const std::vector<AnfNodePtr> &all_nodes) {
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    ValueNodePtr prim_value_node = cnode->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prim_value_node);
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_value_node);
    MS_EXCEPTION_IF_NULL(prim);

    if (IsCommunicationOp(prim) && cnode->in_forward_flag()) {
      MS_EXCEPTION_IF_NULL(prim_value_node->scope());
      MS_LOG(INFO) << "The graph contain communication op: " << prim->name() << ", scope name is "
                   << prim_value_node->scope()->name();
      return true;
    }
  }
  return false;
}

void StepRedistribution(const CNodePtr &node, const OperatorInfoPtr &distribute_operator, const CNodePtr &insert_node,
                        const TensorRedistribution &tensor_redistribution, const CNodePtr &pre_node) {
  MS_EXCEPTION_IF_NULL(node->func_graph());
  FuncGraphManagerPtr manager = node->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodeIndexSet node_set = manager->node_users()[node];
  CNodePtr insert_node_new;

  if (IsPrimitiveCNode(node, prim::kPrimSend)) {
    return;
  }
  if (AnfNodeIsPrimitive(node, MAKE_TUPLE) || AnfNodeIsPrimitive(node, MAKE_LIST)) {
    MS_LOG(INFO) << "No need to insert redistribution op between make_tuple node and the next node";
    return;
  }
  if (IsValueNode<Primitive>(node->input(0))) {
    auto current_value = node->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(current_value);
    PrimitivePtr current_prim = current_value->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(current_prim);
    insert_node_new = ((current_prim->name() == prim::kTupleGetItem) ? node : insert_node);
  } else {
    insert_node_new = insert_node;
  }
  MS_EXCEPTION_IF_NULL(insert_node_new);
  for (auto &node_pair : node_set) {
    CNodePtr use_cnode = node_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(use_cnode);
    if (!IsValueNode<Primitive>(use_cnode->input(0))) {
      StepRedistribution(use_cnode, distribute_operator, insert_node_new, tensor_redistribution, pre_node);
    } else {
      ValueNodePtr prim_anf_node = use_cnode->input(0)->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(prim_anf_node);
      PrimitivePtr node_prim = prim_anf_node->value()->cast<PrimitivePtr>();
      MS_EXCEPTION_IF_NULL(node_prim);
      if ((node_prim->name() == DEPEND && node_pair.second != 1) || node_prim->name() == UPDATESTATE) {
        continue;
      }
      if (IsParallelCareNode(use_cnode) && use_cnode->has_user_data<OperatorInfo>()) {
        Redistribution(node_pair, distribute_operator, insert_node_new, node_pair.second, tensor_redistribution,
                       pre_node);
      } else {
        StepRedistribution(use_cnode, distribute_operator, insert_node_new, tensor_redistribution, pre_node);
      }
    }
  }
}

void SplitTensor(const AnfNodePtr &node, const CNodePtr &next_node, int64_t index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(next_node);
  OperatorInfoPtr op_info = next_node->user_data<OperatorInfo>();
  MS_EXCEPTION_IF_NULL(op_info);

  // If the shape of tensor is [] or [1], no need to split it.
  Shapes shapes = GetNodeShape(node);
  if (shapes.size() != 1) {
    MS_LOG(EXCEPTION) << "Split tensor for " << op_info->name()
                      << ": GetNodeShape for tensor_node, output size is not 1";
  }
  Shape shape = shapes[0];
  std::string shape_str = ShapeToString(shape);
  if (shape.empty() || ((shape.size() == 1) && (shape[0] == 1))) {
    MS_LOG(INFO) << "Split tensor for " << op_info->name() << ": The shape is " << shape_str
                 << ", no need to split it.";
    return;
  }

  MS_LOG(INFO) << "Split tensor for " << op_info->name() << ": The shape of tensor is " << shape_str;

  // extract tensor layout
  if (LongToSize(index - 1) >= op_info->inputs_tensor_info().size()) {
    MS_LOG(EXCEPTION) << "The index is out of range, index is  " << (index - 1) << ", vector size is  "
                      << op_info->inputs_tensor_info().size();
  }
  TensorInfo tensor_info = op_info->inputs_tensor_info()[LongToSize(index - 1)];
  TensorLayout tensor_layout = tensor_info.tensor_layout();

  // Use _GetTensorSlice operator to split the tensor
  FuncGraphPtr func_graph = next_node->func_graph();  // only cnode can get the graph
  MS_EXCEPTION_IF_NULL(func_graph);
  Operator op = CreateGetTensorSliceOp(tensor_layout);
  InsertGetTensorSliceOp(op, next_node, func_graph, index, SPLIT_TENSOR);
  if (!op_info->sub_ops().empty()) {
    auto sub_ops = op_info->sub_ops();
    for (size_t i = 0; i < sub_ops.size(); i++) {
      if (!sub_ops.at(i).empty()) {
        InsertGetTensorSliceOp(sub_ops.at(i).at(0), next_node, func_graph, index, SUB);
      }
    }
  }
}

void SplitTensorList(const AnfNodePtr &node, const CNodePtr &next_node, int index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(next_node);
  if (next_node->inputs().size() != 2 || index != 1) {
    MS_LOG(INFO) << next_node->fullname_with_scope() << " Inputs must have only one input, get "
                 << (next_node->inputs().size() - 1) << " index should be 1, get " << index;
    return;
  }
  OperatorInfoPtr op_info = next_node->user_data<OperatorInfo>();
  MS_EXCEPTION_IF_NULL(op_info);

  std::vector<ValuePtr> inputs_values;
  if (IsValueNode<ValueList>(node)) {
    inputs_values = node->cast<ValueNodePtr>()->value()->cast<ValueListPtr>()->value();
  } else {
    inputs_values = node->cast<ValueNodePtr>()->value()->cast<ValueTuplePtr>()->value();
  }
  if (inputs_values.size() != op_info->inputs_tensor_info().size()) {
    MS_LOG(EXCEPTION) << "The inputs size " << inputs_values.size() << ", is not equal to inputs shape size "
                      << op_info->inputs_tensor_info().size();
  }
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  FuncGraphPtr func_graph = next_node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  ScopePtr scope = next_node->scope();
  MS_EXCEPTION_IF_NULL(scope);
  for (size_t i = 0; i < inputs_values.size(); ++i) {
    auto value_ptr = inputs_values[i];
    auto tensor = value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    TensorInfo tensor_info = op_info->inputs_tensor_info()[i];
    TensorLayout tensor_layout = tensor_info.tensor_layout();
    auto value_node = NewValueNode(value_ptr)->cast<AnfNodePtr>();
    Operator op = CreateGetTensorSliceOp(tensor_layout);
    std::vector<AnfNodePtr> node_input = CreateInput(op, value_node, SPLIT_TENSOR);
    CNodePtr new_node = func_graph->NewCNode(node_input);
    new_node->set_in_forward_flag(true);
    auto new_node_value = node_input[0]->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(new_node_value);
    PrimitivePtr new_node_prim = new_node_value->value()->cast<PrimitivePtr>();
    new_node_prim->set_instance_name(SPLIT_TENSOR);
    new_node_prim->set_attr("keep_value_node_input", MakeValue(true));
    new_node->set_scope(scope);
    node_input[0]->set_scope(scope);
    make_tuple_inputs.push_back(new_node);
  }
  CNodePtr make_tuple = func_graph->NewCNode(make_tuple_inputs);
  manager->Replace(node, make_tuple);
}

void StepSplitTensor(const AnfNodePtr &node, const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodeIndexSet node_set = manager->node_users()[node];
  for (auto &node_pair : node_set) {
    CNodePtr use_cnode = node_pair.first->cast<CNodePtr>();
    if (use_cnode == nullptr || !IsValueNode<Primitive>(use_cnode->input(0))) {
      continue;
    }
    ValueNodePtr prim_anf_node = use_cnode->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prim_anf_node);
    PrimitivePtr use_cnode_prim = prim_anf_node->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(use_cnode_prim);
    if ((use_cnode_prim->name() == DEPEND && node_pair.second != 1) ||
        NO_INPUT_TENSOR_OPS.find(use_cnode_prim->name()) != NO_INPUT_TENSOR_OPS.end()) {
      continue;
    }
    if (IsParallelCareNode(use_cnode)) {
      if (IsValueNode<ValueList>(node) || IsValueNode<ValueTuple>(node)) {
        SplitTensorList(node, use_cnode, node_pair.second);
      } else {
        SplitTensor(node, use_cnode, node_pair.second);
      }
    }
  }
}

void StepReplaceOp(OperatorVector replace_op, const CNodePtr &node) {
  // step1:get graph manager distribute_operator
  OperatorInfoPtr distribute_operator = node->user_data<OperatorInfo>();
  if (distribute_operator == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:AddNode error since distribute_operator is nullptr";
  }
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:AddNode error since manager is nullptr";
  }

  // When reshape(bool), insert cast in the begin and end of op_list to avoid AllGather(bool).
  auto reshape_type_str = node->abstract()->BuildType()->ToString();
  auto replace_op_info = distribute_operator->replace_op_info();
  if (reshape_type_str.find(BOOL) != std::string::npos) {
    auto cast_int = CreateCastOp(kInt32);
    auto cast_bool = CreateCastOp(kBool);
    (void)replace_op.insert(replace_op.begin(), cast_int);
    (void)replace_op.insert(replace_op.end(), cast_bool);
    (void)replace_op_info.insert(replace_op_info.begin(), {false, 1});
    (void)replace_op_info.insert(replace_op_info.end(), {false, 1});
  }

  // step2:traverse op_list and insert node
  std::reverse(replace_op.begin(), replace_op.end());
  std::reverse(replace_op_info.begin(), replace_op_info.end());
  if (!replace_op_info.empty() && replace_op_info.size() != replace_op.size()) {
    MS_LOG(EXCEPTION) << "replace_op_info is not empty and size not equal to replace_op!";
  }
  bool replace_op_info_flag = !replace_op_info.empty();
  for (size_t index = 0; index < replace_op.size(); ++index) {
    std::string instance_name = CreateInstanceName(node, index);
    std::vector<AnfNodePtr> replace_input;
    if (index != replace_op.size() - 1) {
      replace_input = CreateInput(replace_op[index], node, instance_name);
    } else {
      replace_input = ReplaceOpInput(replace_op[index], instance_name, node);
    }
    CNodePtr replace_node = func_graph->NewCNode(replace_input);
    MS_EXCEPTION_IF_NULL(replace_node);
    ScopePtr scope = node->scope();
    MS_EXCEPTION_IF_NULL(scope);
    replace_node->set_scope(scope);
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(replace_node->input(0));
    PrimitivePtr origin_prim = GetValueNode<PrimitivePtr>(node->input(0));
    SetUserAttrs(origin_prim->attrs(), prim);
    auto origin_prim_attrs = origin_prim->attrs();
    if (origin_prim_attrs.find(RECOMPUTE_COMM_OP) != origin_prim_attrs.end() &&
        !GetValue<bool>(origin_prim_attrs[RECOMPUTE_COMM_OP]) &&
        COMMUNICATION_OPS.find(prim->name()) != COMMUNICATION_OPS.end()) {
      MS_LOG(INFO) << "The redistribution node in reshape would not be recomputed.";
      prim->set_attr("recompute", MakeValue(false));
    }
    if (index == replace_op.size() - 1) {
      replace_node->set_user_data<OperatorInfo>(node->user_data<OperatorInfo>());
      replace_node->set_primal_attrs(node->primal_attrs());
    }
    replace_node->set_in_forward_flag(true);
    replace_input[0]->set_scope(scope);
    if (replace_op_info_flag && replace_op_info[index].first) {
      auto new_cnode = InsertMakeTuple(replace_node, replace_op_info[index].second, func_graph);
      new_cnode->set_primal_attrs(node->primal_attrs());
      (void)manager->Replace(node, new_cnode);  // using Replace function to insert node
    } else {
      (void)manager->Replace(node, replace_node);  // using Replace function to insert node
    }
  }
  MS_LOG(INFO) << "Insert ReplaceOp success for " << distribute_operator->name();
}

void StepReplaceGraph(const ReplaceGraphPtr &replace_graph, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(replace_graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(replace_graph->second);
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:AddNode error since manager is nullptr";
  }
  // Solve the input order
  // For example input_node:{segment_sum:1, segment_sum:2, gahter:2}
  // The Original code here will bind the all operations to the first inputs of these operatos
  // However, the segment_sum operation needs two inputs, To solve this
  // We maintain a dict to count the times of the same operations,
  // and bind the inputs according to the times of the op appears.
  mindspore::HashMap<AnfNodePtr, int> input_map = {};
  static int appear_count = 0;
  for (auto &replace_input : replace_graph->first) {
    auto pre_node = node->input(LongToSize(replace_input.second));

    auto it = input_map.find(replace_input.first);
    if (it != input_map.end()) {
      appear_count = 1 + it->second;
    } else {
      appear_count = 1;
    }
    auto replace_input_cnode = replace_input.first->cast<CNodePtr>();
    size_t inputs_size = replace_input_cnode->inputs().size();
    while (IntToSize(appear_count) < inputs_size && replace_input_cnode->input(appear_count)->func_graph() != nullptr) {
      ++appear_count;
    }
    if (IntToSize(appear_count) >= inputs_size) {
      MS_LOG(EXCEPTION) << "No replaceable virtual_input_node";
    }
    input_map[replace_input.first] = appear_count;
    manager->SetEdge(replace_input.first, appear_count, pre_node);
  }
  //  "(void)manager->Replace(replace_graph->first, pre_node);" can not be called
  auto replace_output = replace_graph->second->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(replace_output);
  replace_output->set_primal_attrs(node->primal_attrs());
  (void)manager->Replace(node, replace_output);
}

int64_t GetTupleGetItemIndex(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().size() != 3) {
    MS_LOG(EXCEPTION) << cnode->ToString() << " size( " << cnode->inputs().size() << " ) is not 3";
  }

  if (!cnode->input(TUPLE_GETITEM_INDEX_POS)->isa<ValueNode>()) {
    MS_LOG(EXCEPTION) << "The index of tuple getitem is not a value node";
  }

  ValuePtr tuple_index_value = GetValueNode(cnode->input(TUPLE_GETITEM_INDEX_POS));
  MS_EXCEPTION_IF_NULL(tuple_index_value);
  if (!tuple_index_value->isa<Int64Imm>()) {
    MS_LOG(EXCEPTION) << "The index of tuple getitem is not int32";
  }
  return tuple_index_value->cast<Int64ImmPtr>()->value();
}

void InsertVirtualDivOp(const VirtualDivOp &virtual_div_op, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  size_t node_size = node->inputs().size();
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  if (IsSomePrimitive(node, DROPOUT_DO_MASK)) {
    MS_LOG(INFO) << "Handle dropout do mask, only insert the virtual div to input[0]";
    node_size = 2;
  }

  for (size_t index = 1; index < node_size; ++index) {
    AnfNodePtr input = node->input(index);
    MS_EXCEPTION_IF_NULL(input);
    // if it is not a tensor, continue
    if ((!input->isa<CNode>() && !input->isa<Parameter>()) || HasAbstractMonad(input)) {
      MS_LOG(INFO) << "insert div op: the index  " << index << "  is not tensor, skip";
      continue;
    }

    for (size_t pos = 0; pos < virtual_div_op.size(); ++pos) {
      std::string instance_name = CreateInstanceName(node, pos);
      InsertNode(virtual_div_op[pos], node, index, node->input(index), func_graph, instance_name);
    }
    MS_LOG(INFO) << "insert div op for input index  " << index << "  of node";
  }
}

void InsertRealDivOpToNodeInput(const CNodePtr &node, int64_t scale, const string &instance_name) {
  MS_EXCEPTION_IF_NULL(node);
  if (scale == 0) {
    MS_LOG(EXCEPTION) << "Find the scale value is 0, you should check the mirror operators's group size.";
  }
  size_t node_size = node->inputs().size();
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  // instance the real div operator
  Operator div_op = CreateDivOp(scale);

  // Insert it as the input of the node
  for (size_t index = 1; index < node_size; ++index) {
    AnfNodePtr input = node->input(index);
    MS_EXCEPTION_IF_NULL(input);
    // if it is not a tensor, continue
    if ((!input->isa<CNode>() && !input->isa<Parameter>()) || HasAbstractMonad(input)) {
      continue;
    }
    InsertNode(div_op, node, index, node->input(index), func_graph, instance_name);
  }
}

void InsertAllReduceToNodeInput(const CNodePtr &node, const std::string &group, const std::string &instance_name) {
  MS_EXCEPTION_IF_NULL(node);
  size_t node_size = node->inputs().size();
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  // instance the real div operator
  CheckGlobalDeviceManager();
  Operator allreduce_op = CreateAllReduceOp(REDUCE_OP_SUM, group);

  // Insert it as the input of the node
  for (size_t index = 1; index < node_size; ++index) {
    AnfNodePtr input = node->input(index);
    MS_EXCEPTION_IF_NULL(input);
    // if it is not a tensor, continue
    if ((!input->isa<CNode>() && !input->isa<Parameter>()) || HasAbstractMonad(input)) {
      continue;
    }

    InsertNode(allreduce_op, node, index, node->input(index), func_graph, instance_name);
  }
}

FuncGraphPtr PynativeParallelGraph(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes) {
  FuncGraphPtr real_graph = root;
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    auto expect_shard_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (expect_shard_prim->name() != SHARD) {
      continue;
    }
    real_graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
  }
  return real_graph;
}

void InsertVirtualOutput(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes) {
  std::vector<std::string> last_forward_node_ids;
  std::vector<size_t> last_indexs;
  auto real_graph = PynativeParallelGraph(root, all_nodes);
  FindLastNodesUniqueId(real_graph, &last_forward_node_ids, &last_indexs);
  MS_LOG(INFO) << "there are " << last_forward_node_ids.size() << " output nodes in eval/predict";
  for (auto &node : all_nodes) {
    // here insert virtualoutput node
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    auto last_node_iter = std::find(last_forward_node_ids.begin(), last_forward_node_ids.end(), cnode->UniqueId());
    if (last_node_iter == last_forward_node_ids.end()) {
      continue;
    }
    for (size_t last_node_index = 0; last_node_index < last_forward_node_ids.size(); ++last_node_index) {
      if (last_forward_node_ids[last_node_index] != cnode->UniqueId()) {
        continue;
      }
      MS_LOG(INFO) << "find last node: " << cnode->fullname_with_scope() << ", the parallel care node is: "
                   << cnode->input(last_indexs[last_node_index])->fullname_with_scope();
      if (IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem)) {
        FuncGraphManagerPtr manager = cnode->func_graph()->manager();
        MS_EXCEPTION_IF_NULL(manager);
        auto node_pair = manager->node_users()[cnode].front();
        if (!node_pair.first->isa<CNode>()) {
          MS_LOG(EXCEPTION) << "the output of tuple_get_item is not a cnode";
        }
        cnode = node_pair.first->cast<CNodePtr>();
        last_indexs[last_node_index] = IntToSize(node_pair.second);
      }
      auto pre_node = cnode->input(last_indexs[last_node_index]);
      Shapes shape_outputs = GetNodeShape(pre_node);
      if (shape_outputs[0].empty()) {
        continue;
      }
      FuncGraphPtr func_graph = node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      OperatorParams params;
      OperatorAttrs attrs;
      OperatorArgs args = std::make_pair(attrs, params);
      Operator op = std::make_pair(VIRTUAL_OUTPUT, args);
      InsertNode(op, cnode, last_indexs[last_node_index], pre_node, func_graph, VIRTUAL_OUTPUT);
      auto virtual_output_node = cnode->input(last_indexs[last_node_index]);
      AbstractBasePtr virtual_output_abstract = pre_node->abstract()->Clone();
      std::shared_ptr<abstract::BaseShape> virtual_output_shape = std::make_shared<abstract::Shape>(shape_outputs[0]);
      virtual_output_abstract->set_shape(virtual_output_shape);
      virtual_output_node->set_abstract(virtual_output_abstract);
    }
  }
}

// only used for FindCNode
CNodePtr SkipTrivialNodesMoveDown(const FuncGraphManagerPtr &manager, CNodePtr node) {
  MS_EXCEPTION_IF_NULL(node);
  while (IsInTrivialNodeList(node) || IsSomePrimitive(node, LOAD)) {
    node = manager->node_users()[node].begin()->first->cast<CNodePtr>();
  }
  return node;
}

std::pair<bool, CNodePtr> FindCNode(const AnfNodePtr &anode, const std::string &name, const FuncGraphPtr &func_graph,
                                    size_t max_depth) {
  MS_EXCEPTION_IF_NULL(anode);
  MS_EXCEPTION_IF_NULL(anode->func_graph());
  FuncGraphManagerPtr manager = anode->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (max_depth > MAX_RECURSIVE_DEPTH) {
    MS_LOG(EXCEPTION) << "Recursive call is larger than 100000.";
  }
  AnfNodeIndexSet node_set = manager->node_users()[anode];
  bool result = false;
  CNodePtr cnode_return = nullptr;
  for (auto &node_pair : node_set) {
    CNodePtr use_apply = node_pair.first->cast<CNodePtr>();
    if (use_apply == nullptr || !IsValueNode<Primitive>(use_apply->input(0))) {
      continue;
    }
    use_apply = SkipTrivialNodesMoveDown(manager, use_apply);
    if (use_apply == nullptr || !IsValueNode<Primitive>(use_apply->input(0))) {
      continue;
    }
    ValueNodePtr prim_anf_node = use_apply->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prim_anf_node);
    PrimitivePtr node_prim = prim_anf_node->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(node_prim);
    if (node_prim->name() == name && node_pair.second == 1) {
      if (use_apply->func_graph() == func_graph) {
        result = true;
        cnode_return = use_apply;
        MS_LOG(INFO) << "Find Primitive " << name << " in the same func_graph";
        continue;
      }
      MS_LOG(INFO) << "Find Primitive " << name << " in different func_graph";
    }
    if (ParallelContext::GetInstance()->enable_parallel_optimizer() && IsInAllGatherNodeList(use_apply)) {
      return FindCNode(node_pair.first, name, func_graph, max_depth + 1);
    }
  }
  return std::make_pair(result, cnode_return);
}

bool InsertMirrorBeforeCast(const CNodePtr &node, size_t index) {
  // only if gradient_fp32_sync is true, pre node is cast and type is not float32 return true
  if (!ParallelContext::GetInstance()->gradient_fp32_sync()) {
    return false;
  }
  auto pre_node = node->input(index);
  MS_EXCEPTION_IF_NULL(pre_node);
  auto cnode = pre_node->cast<CNodePtr>();
  if (cnode == nullptr || !IsValueNode<Primitive>(cnode->input(0))) {
    return false;
  }
  if (ParallelContext::GetInstance()->enable_parallel_optimizer() && IsInAllGatherNodeList(cnode)) {
    pre_node = cnode->input(1);
  }
  if (!IsPrimitiveCNode(pre_node, prim::kPrimCast)) {
    return false;
  }
  auto node_type = pre_node->Type();
  MS_EXCEPTION_IF_NULL(node_type);
  if (!node_type->isa<mindspore::TensorType>()) {
    MS_LOG(EXCEPTION) << "Unknown type.";
  }
  auto input_element_type = node_type->cast<mindspore::TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(input_element_type);
  auto type_id = input_element_type->type_id();

  return (type_id != kNumberTypeFloat32);
}

static bool CheckInsertMirrorOps(const MirrorOps &mirror_ops, const CNodePtr &node, size_t node_size) {
  if (IsPrimitiveCNode(node, prim::kPrimSend)) {
    return true;
  }
  constexpr size_t kSingleArgCNodeSize = 2;
  if ((node->inputs().size() == kSingleArgCNodeSize) && (IsValueNode<ValueSequence>(node->input(1)))) {
    MS_LOG(INFO) << "Input is ValueList, skip it.";
    return false;
  }

  if ((node->inputs().size() == kSingleArgCNodeSize) &&
      (AnfNodeIsPrimitive(node->input(1), MAKE_TUPLE) || AnfNodeIsPrimitive(node->input(1), MAKE_LIST))) {
    MS_LOG(INFO) << "The mirror for " << GetPrimName(node) << " has handle by make_tuple node";
    return false;
  }

  if (mirror_ops.size() != node_size - 1) {
    MS_LOG(EXCEPTION) << "Mirrorops's size is wrong! mirror_ops size is " << mirror_ops.size() << ", node_size is "
                      << (node_size - 1);
  }
  return true;
}

// only used for InsertMirrorOps
CNodePtr SkipTrivialNodesMoveUp(CNodePtr node) {
  MS_EXCEPTION_IF_NULL(node);
  while (!IsSomePrimitive(node, LOAD)) {
    if (IsInTrivialNodeList(node) || IsInAllGatherNodeList(node)) {
      node = node->input(1)->cast<CNodePtr>();
    }
  }
  auto prev_node = node->input(1)->cast<CNodePtr>();
  if (prev_node != nullptr) {
    if (IsSomePrimitive(prev_node, DEPEND)) {
      auto prev_prev_node = prev_node->input(1)->cast<CNodePtr>();
      if (IsSomePrimitive(node, LOAD)) {
        node = prev_prev_node;
        MS_LOG(INFO) << "Moving to the Load node before Depend node.";
      }
    }
  }
  return node;
}

std::string MirrorOpName() {
  int64_t grad_accumulation_step = ParallelContext::GetInstance()->grad_accumulation_step();
  int64_t split_stage_num = ParallelContext::GetInstance()->pipeline_stage_split_num();
  std::string mirror_op_name;
  if (grad_accumulation_step > 1) {
    mirror_op_name = MIRROR_MINI_STEP_OPERATOR;
  } else if (split_stage_num > 1) {
    mirror_op_name = MIRROR_MICRO_STEP_OPERATOR;
  } else {
    mirror_op_name = MIRROR_OPERATOR;
  }
  return mirror_op_name;
}

static void DoInsertMirrorOps(const FuncGraphPtr &root, const MirrorOps &mirror_ops, const CNodePtr &node,
                              size_t node_size) {
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  for (size_t index = 1; index < node_size; ++index) {
    OperatorVector backward_op = mirror_ops[index - 1];
    if (IsPrimitiveCNode(node, prim::kPrimSend)) {
      auto param_index = GetValue<int>(node->GetPrimalAttr(PARAM_INDEX));
      backward_op = mirror_ops[IntToSize(param_index)];
    }
    if (backward_op.empty()) {
      continue;
    }
    std::pair<AnfNodePtr, bool> param_node_pair = FindParameter(node->input(index), func_graph);
    if (!param_node_pair.first) {
      continue;
    }

    auto param_ptr = param_node_pair.first->cast<ParameterPtr>();
    std::string param_name;
    bool is_shared_param = false;
    if (param_ptr) {
      param_name = param_ptr->name();
      if (!param_ptr->param_info() || !param_ptr->param_info()->requires_grad()) {
        MS_LOG(INFO) << param_name << " do not need gradient. Skip inserting mirror.";
        continue;
      }
      std::string opt_shard_mirror_group;
      if (param_ptr->user_data<TensorLayout>()) {
        opt_shard_mirror_group = param_ptr->user_data<TensorLayout>()->opt_shard_mirror_group();
        is_shared_param = param_ptr->user_data<TensorLayout>()->is_shared_param();
      }
      if (!opt_shard_mirror_group.empty()) {
        // mirror ops is covered in not fully use opt shard case
        uint32_t group_rank_size = 0;
        if (!CommManager::GetInstance().GetRankSize(opt_shard_mirror_group, &group_rank_size)) {
          MS_LOG(EXCEPTION) << "Got the group size from the group " << opt_shard_mirror_group << " failed";
        }
        backward_op = CreateMirrorOps(opt_shard_mirror_group, static_cast<size_t>(group_rank_size));
      }
    }
    // not a RefKey
    std::string mirror_op_name = MirrorOpName();
    AnfNodePtr pre_node = node->input(index);
    if (!param_node_pair.second) {
      auto next_cnode = FindCNode(param_node_pair.first, mirror_op_name, func_graph, 0);
      // if there is already a MirrorOp in the same graph, use MirrorOp CNode as a input instead
      if (next_cnode.first) {
        MS_EXCEPTION_IF_NULL(next_cnode.second);
        // assume Load is inserted next to parameter
        // skip Load moving up and insert mirror next to the parameter
        if (pre_node->cast<CNodePtr>()) {
          CNodePtr load_node = SkipTrivialNodesMoveUp(node->input(index)->cast<CNodePtr>());
          manager->SetEdge(load_node, 1, next_cnode.second);
        } else {
          manager->SetEdge(node, static_cast<int>(index), next_cnode.second);
        }
        MS_LOG(INFO) << "Find parameter " << param_name << " for node " << GetPrimName(node->cast<CNodePtr>())
                     << " and share the mirror.";
        continue;
      }
    }
    // if the parameter found is a RefKey, or no MirrorOp is found in the same graph, insert a new MirrorOp
    // only one MirrorOp in backward_op
    if (backward_op.size() != 1) {
      MS_LOG(EXCEPTION) << "backward_op size must be 1, real is " << backward_op.size();
    }
    auto op = backward_op[0];
    if (pre_node->cast<CNodePtr>() && (InsertMirrorBeforeCast(node, index) || is_shared_param)) {
      // assume Load is inserted next to parameter
      // skip Load moving up and insert mirror next to the parameter
      CNodePtr load_node = SkipTrivialNodesMoveUp(pre_node->cast<CNodePtr>());
      InsertNode(op, load_node, 1, load_node->input(1), func_graph, mirror_op_name, param_name, root);
      auto comm_op = load_node->input(1)->cast<CNodePtr>();
      // add fusion flag
      AddCommOpFusionType(comm_op, param_node_pair.first);
      MS_LOG(INFO) << "Find parameter " << param_name << " for node " << GetPrimName(node->cast<CNodePtr>())
                   << " and insert mirror before Load";
      AddCommOpParamFlag(comm_op);
      continue;
    }
    InsertNode(op, node, index, pre_node, func_graph, mirror_op_name, param_name, root);
    MS_LOG(INFO) << "Find parameter " << param_name << " for node " << GetPrimName(node->cast<CNodePtr>())
                 << " and insert mirror before the node";
    auto comm_op = node->input(index)->cast<CNodePtr>();
    // add fusion flag
    // pipeline mirror would not be set, which should be supported later
    AddCommOpFusionType(comm_op, param_node_pair.first);
    AddCommOpParamFlag(comm_op);
  }
}

void InsertMirrorOps(const FuncGraphPtr &root, const MirrorOps &mirror_ops, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  size_t node_size = node->inputs().size();
  for (auto input : node->inputs()) {
    if (HasAbstractMonad(input)) {
      node_size--;
    }
  }

  if (!CheckInsertMirrorOps(mirror_ops, node, node_size)) {
    return;
  }

  DoInsertMirrorOps(root, mirror_ops, node, node_size);
}

void BackwardCommunication(const FuncGraphPtr &root, const OperatorInfoPtr &distribute_operator, const CNodePtr &node,
                           const std::vector<std::pair<CNodePtr, LossNodeInfo>> &sens_loss_pairs) {
  MS_EXCEPTION_IF_NULL(distribute_operator);
  MS_EXCEPTION_IF_NULL(node);

  if (IsPrimitiveCNode(node, prim::kPrimReceive)) {
    return;
  }
  bool is_loss_cnode =
    std::any_of(sens_loss_pairs.begin(), sens_loss_pairs.end(),
                [node](const std::pair<CNodePtr, LossNodeInfo> &element) { return element.second.loss_node == node; });

  MirrorOps mirror_ops = distribute_operator->mirror_ops();
  VirtualDivOp virtual_div_op = distribute_operator->virtual_div_op();
  // insert mirror op
  if (!mirror_ops.empty()) {
    MS_LOG(INFO) << "insert mirror op for " << distribute_operator->name();
    InsertMirrorOps(root, mirror_ops, node);
  }
  // insert virtual div op
  if (!virtual_div_op.empty() && is_loss_cnode && IsLastStage()) {
    MS_LOG(INFO) << "insert virtual div op for " << distribute_operator->name();
    InsertVirtualDivOp(virtual_div_op, node);
  }
}

std::string GetDisOpName(const std::string &prim_name) {
  std::string op_name = prim_name;
  if (!prim_name.empty() && (prim_name[0] == '_')) {
    op_name = prim_name.substr(1);
  }
  return op_name + "Info";
}

OperatorInfoPtr OperatorInstanceByName(const std::string &name, const PrimitiveAttrs &attrs,
                                       const std::vector<Shapes> &shape_list) {
  if (shape_list.size() != 2) {
    MS_LOG(ERROR) << "The size of shape list is not 2";
    return nullptr;
  }
  if (name.length() == 0) {
    MS_LOG(EXCEPTION) << "Length of name is zero!";
  }
  std::string distribute_opname = GetDisOpName(name);
  OperatorInfoPtr operator_ =
    (OperatorInfoPtr)DynCreator::Instance().Create(distribute_opname, shape_list[0], shape_list[1], attrs, TOTAL_OPS);
  if (operator_ == nullptr) {
    MS_LOG(INFO) << "Create " << name << " failed";
    return nullptr;
  }
  std::string origin_name = operator_->name();
  operator_->set_name(origin_name + std::to_string(TOTAL_OPS));
  MS_LOG(INFO) << "Successfully created operator " << origin_name;
  ++TOTAL_OPS;
  return operator_;
}

OperatorInfoPtr OperatorInstance(const PrimitivePtr &prim, const PrimitiveAttrs &attrs,
                                 const std::vector<Shapes> &shape_list) {
  MS_EXCEPTION_IF_NULL(prim);
  OperatorInfoPtr operator_ = OperatorInstanceByName(prim->name(), attrs, shape_list);
  if (operator_ == nullptr) {
    if (IsInBatchParallelBlackList(prim)) {
      MS_LOG(EXCEPTION) << "Operator " << prim->name() << " is not supported yet in auto parallel mode.";
    }
    MS_LOG(INFO) << "Create " << prim->name() << " failed, use batch parallel";
    operator_ = OperatorInstanceByName(BATCH_PARALLEL, attrs, shape_list);
    MS_EXCEPTION_IF_NULL(operator_);
  }
  return operator_;
}

OperatorInfoPtr NewOperatorInstance(const PrimitivePtr &prim, const PrimitiveAttrs &attrs,
                                    std::vector<Shapes> shape_list) {
  OperatorInfoPtr operator_ = OperatorInstance(prim, attrs, shape_list);
  for (size_t i = 0; i < shape_list[0].size(); ++i) {
    MS_LOG(INFO) << "No:  " << i << "  input's shape: " << ShapeToString(shape_list[0][i]);
  }
  return operator_;
}

StrategyPtr ExtractStrategy(const ValuePtr &stra) {
  if (stra == nullptr) {
    return nullptr;
  }

  auto var = stra->cast<ValueTuplePtr>();
  if (var == nullptr) {
    return nullptr;
  }

  StrategyPtr strategyPtr;
  int64_t stage_id = g_device_manager->stage_id();

  MS_LOG(INFO) << "Extract information: strategy " << stra->ToString();
  if (var->size() > 0) {
    std::vector<ValuePtr> elements = var->value();
    Strategys strategy;
    for (uint64_t index = 0; index < elements.size(); ++index) {
      Dimensions dim;
      if (elements[index]->isa<ValueSequence>()) {
        auto value_tuple = elements[index]->cast<ValueTuplePtr>();
        std::vector<ValuePtr> value_vector = value_tuple->value();
        (void)std::transform(value_vector.begin(), value_vector.end(), std::back_inserter(dim),
                             [](const ValuePtr &value) { return static_cast<int64_t>(GetValue<int64_t>(value)); });
        strategy.push_back(dim);
      } else {
        MS_LOG(EXCEPTION) << "Failure: Strategy's format is wrong! Need ValueSequence";
      }
    }
    if (strategy.empty()) {
      MS_LOG(EXCEPTION) << "ExtractStrategy: failed to extract strategy";
    }
    strategyPtr = NewStrategy(stage_id, strategy);
  }

  return strategyPtr;
}

Shapes GetRefKeyNodeShape(const AnfNodePtr &node, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);

  std::vector<AnfNodePtr> parameters = FindParameterByRefKeyNode(node, func_graph);
  if (parameters.size() != 1) {
    MS_LOG(EXCEPTION) << "Find parameter by ref key node failed";
  }

  Shapes input_shapes;
  input_shapes = GetNodeShape(parameters[0]);
  if (input_shapes.size() != 1) {
    MS_LOG(EXCEPTION) << "Get input shape failed";
  }

  MS_LOG(INFO) << "The parameter shape is " << ShapeToString(input_shapes[0]);
  return input_shapes;
}

std::vector<Shapes> ExtractShape(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  Shapes shape_inputs, shape_outputs;
  std::vector<Shapes> shape_all;
  std::vector<AnfNodePtr> all_inputs = node->inputs();

  size_t inputs_size = all_inputs.size();
  for (size_t i = 1; i < inputs_size; ++i) {
    Shapes input_shapes;
    AnfNodePtr input = all_inputs[i];
    if (HasAbstractMonad(input)) {
      continue;
    }
    if (IsValueNode<RefKey>(input)) {
      auto func_graph = node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      std::vector<AnfNodePtr> parameters = FindParameterByRefKeyNode(input, func_graph);
      if (parameters.size() != 1) {
        MS_LOG(EXCEPTION) << "Find parameter by ref key node failed";
      }
      std::pair<AnfNodePtr, int64_t> node_pair = std::make_pair(node, SizeToLong(i));
      g_RefMap[parameters[0]] = node_pair;
      input_shapes = GetRefKeyNodeShape(input, func_graph);
    } else if (input->isa<CNode>() || IsValueNode<Tensor>(input) || input->isa<Parameter>() ||
               ((IsValueNode<ValueList>(input) || IsValueNode<ValueTuple>(input)) && (inputs_size == 2))) {
      input_shapes = GetNodeShape(input);
    } else {
      continue;
    }
    if (input_shapes.size() != 1) {
      if (inputs_size == 2) {  // like concat
        shape_inputs = input_shapes;
        break;
      } else {
        MS_LOG(EXCEPTION) << "ExtractShape: Get input shape failed";
      }
    }
    shape_inputs.push_back(input_shapes[0]);
  }
  shape_all.push_back(shape_inputs);
  // extract out shape
  shape_outputs = GetNodeShape(node);
  shape_all.push_back(shape_outputs);
  return shape_all;
}

std::pair<AnfNodePtr, int64_t> FindParallelCareNode(const AnfNodePtr &node, int32_t recursion_num) {
  if (recursion_num >= RECURSION_LIMIT) {
    return std::make_pair(nullptr, 0);
  }

  MS_EXCEPTION_IF_NULL(node);
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodeIndexSet node_set = manager->node_users()[node];
  for (auto &node_pair : node_set) {
    CNodePtr cnode = node_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    ValueNodePtr prim_node_anf = cnode->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prim_node_anf);
    PrimitivePtr node_prim = prim_node_anf->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(node_prim);
    if ((node_prim->name() == DEPEND && node_pair.second != 1) || IsPrimitiveCNode(cnode, prim::kPrimReceive) ||
        IsPrimitiveCNode(cnode, prim::kPrimSend)) {
      continue;
    }
    if (IsParallelCareNode(cnode) && cnode->has_user_data<OperatorInfo>()) {
      return node_pair;
    } else {
      auto tmp_pair = FindParallelCareNode(node_pair.first, recursion_num + 1);
      if (tmp_pair.first != nullptr) {
        return tmp_pair;
      }
    }
  }
  return std::make_pair(nullptr, 0);
}

std::pair<AnfNodePtr, int64_t> FindSubGraph(const FuncGraphPtr &graph, const AnfNodePtr &parameter) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(parameter);
  FuncGraphManagerPtr manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::pair<AnfNodePtr, int64_t> prim_anf_node_pair = FindParallelCareNode(parameter, 0);
  if (prim_anf_node_pair.first != nullptr) {
    return prim_anf_node_pair;
  } else {
    AnfNodeIndexSet param_sub_set = manager->node_users()[parameter];
    for (auto &param_pair : param_sub_set) {
      CNodePtr param_cnode = param_pair.first->cast<CNodePtr>();
      AnfNodePtr graph_value_node;
      if (param_cnode->input(0)->isa<CNode>()) {
        graph_value_node = param_cnode->input(0)->cast<CNodePtr>()->input(1);
      } else {
        graph_value_node = param_cnode->input(0);
      }
      if (!IsValueNode<FuncGraph>(graph_value_node)) {
        continue;
      }
      FuncGraphPtr graph_sub = GetValueNode<FuncGraphPtr>(graph_value_node);
      auto parameters = graph_sub->parameters();
      if (LongToSize(param_pair.second - 1) >= parameters.size()) {
        MS_LOG(EXCEPTION) << "The index is out of range, index is: " << (param_pair.second - 1) << ", vector size is "
                          << parameters.size();
      }
      std::pair<AnfNodePtr, int64_t> res = FindSubGraph(graph_sub, parameters[LongToSize(param_pair.second - 1)]);
      if (res.first != nullptr) {
        return res;
      }
    }
  }
  return std::make_pair(nullptr, 0);
}

CNodePtr InsertAllGatherAfterCast(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  // skip Load moving down and assume it only has one node user
  CNodePtr res = cnode;
  if (IsSomePrimitive(res, LOAD)) {
    res = manager->node_users()[cnode].begin()->first->cast<CNodePtr>();
  }
  // return true only if cnode is Cast from fp32 to fp16
  if (!IsSomePrimitive(res, CAST)) {
    return nullptr;
  }
  auto node_type = res->Type();
  MS_EXCEPTION_IF_NULL(node_type);
  if (!node_type->isa<mindspore::TensorType>()) {
    MS_LOG(EXCEPTION) << "Unknown type.";
  }
  auto input_element_type = node_type->cast<mindspore::TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(input_element_type);
  auto type_id = input_element_type->type_id();

  if (type_id != kNumberTypeFloat32) {
    return res;
  } else {
    return nullptr;
  }
}

static void InsertAllGatherOp(const FuncGraphPtr &root, const std::string &group, const std::pair<AnfNodePtr, int> &res,
                              const AnfNodePtr &node, const std::string &op_name, bool is_shared_param) {
  MS_EXCEPTION_IF_NULL(res.first);
  MS_EXCEPTION_IF_NULL(node);
  bool grad_accumulation_shard = ParallelContext::GetInstance()->grad_accumulation_shard();
  auto cnode = res.first->cast<CNodePtr>();
  auto graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto cnode_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_EXCEPTION_IF_NULL(cnode_prim);
  Operator op;
  CNodePtr allgather;
  auto param_name = node->cast<ParameterPtr>()->name();
  if (op_name == MINI_STEP_ALL_GATHER) {
    op = CreateMiniStepAllGatherOp(group);
  } else if (op_name == MICRO_STEP_ALL_GATHER) {
    op = CreateMicroStepAllGatherOp(group);
  } else {
    op = CreateAllGatherOp(group);
  }
  CNodePtr cast_node = InsertAllGatherAfterCast(cnode);
  std::string opt_shard_mirror_group;
  auto param_ptr = node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(param_ptr);
  if (param_ptr->user_data<TensorLayout>()) {
    opt_shard_mirror_group = param_ptr->user_data<TensorLayout>()->opt_shard_mirror_group();
  }
  if (!is_shared_param && cast_node) {
    allgather = ReplaceNode(op, cast_node, graph, PARALLEL_OPTIMIZER_ALLGATHER_NOT_COMPUTE, param_name, root);
    MS_LOG(INFO) << "Parallel optimizer is applied before Cast for " << param_name;
  } else {
    auto pre_node = node;
    AnfNodePtr pre_node_ = node;
    auto node_user_map = manager->node_users();
    TypePtr next_node_dtype = FindChildCastWithFP32ToFP16(cnode, node_user_map);
    if (next_node_dtype) {
      MS_LOG(INFO) << "Inserting Cast from float32 to float16 for node " << node->fullname_with_scope() << " for saving"
                   << " communication.";
      pre_node_ = CreateFP16Cast(cnode, pre_node, next_node_dtype);
    }
    InsertNode(op, cnode, IntToSize(res.second), pre_node_, graph, PARALLEL_OPTIMIZER_ALLGATHER_NOT_COMPUTE, param_name,
               root);
    allgather = cnode->input(IntToSize(res.second))->cast<CNodePtr>();
    MS_LOG(INFO) << "Parallel optimizer is applied before " << GetPrimName(cnode) << " for " << param_name;
  }
  // add fusion flag
  AddCommOpFusionType(allgather, node);
  // add gradients mean
  AddCommOpMeanFlag(allgather);
  if (op_name == MICRO_STEP_ALL_GATHER) {
    // When grad_accumulation_shard is enabled, the ReduceScatter is inserted at each micro step
    // so no need to do backward for the micro_step_allgather
    AddCommOpMirrorFlag(allgather, !grad_accumulation_shard);
  } else if (op_name == MINI_STEP_ALL_GATHER) {
    // We need to manually set the add_accu to be false if it's father node is MirrorMiniStep
    bool add_accu = root->has_flag(kAccumulation);
    bool is_with_mirror = opt_shard_mirror_group.size() > 1;
    AddCommOpAddAccuFlag(allgather, !add_accu && !is_with_mirror);
    AddCommOpMirrorFlag(allgather, grad_accumulation_shard || !add_accu);
  }
}

static void ApplyParallelOptOnParam(const FuncGraphPtr &root, const AnfNodePtr &parameter,
                                    const std::string &opt_shard_group) {
  if (opt_shard_group.empty()) {
    return;
  }

  // set all gather type
  MS_EXCEPTION_IF_NULL(parameter);
  int64_t grad_accumulation_step = ParallelContext::GetInstance()->grad_accumulation_step();
  int32_t split_stage_num = ParallelContext::GetInstance()->pipeline_stage_split_num();
  std::string op_name;
  if (grad_accumulation_step > 1) {
    op_name = MINI_STEP_ALL_GATHER;
  } else if (split_stage_num > 1) {
    op_name = MICRO_STEP_ALL_GATHER;
  } else {
    op_name = ALL_GATHER;
  }

  // insert all gather
  FuncGraphManagerPtr manager = root->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto param_sub_set = manager->node_users()[parameter];
  bool insert_flag = false;
  for (auto &param_pair : param_sub_set) {
    auto cnode = param_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->in_forward_flag() && !IsPrimitiveCNode(cnode, prim::kPrimReceive) &&
        !IsPrimitiveCNode(cnode, prim::kPrimDepend)) {
      OperatorInfoPtr distribute_operator = cnode->user_data<OperatorInfo>();
      if (distribute_operator == nullptr) {
        MS_LOG(DEBUG) << "Parallel optimizer: " << GetPrimName(cnode) << " 's OperatorInfoPtr is nullptr";
      } else if (IntToSize(param_pair.second - 1) >= distribute_operator->inputs_tensor_info().size()) {
        MS_LOG(EXCEPTION) << "The index is out of range, index is  " << (param_pair.second - 1) << ", vector size is  "
                          << distribute_operator->inputs_tensor_info().size();
      }

      if (insert_flag) {
        // if there are multiple node users, they share one same allgather
        auto next_cnode = FindCNode(parameter, op_name, cnode->func_graph(), 0);
        if (next_cnode.first) {
          manager->SetEdge(cnode, param_pair.second, next_cnode.second);
          MS_LOG(INFO) << "Parallel optimizer is shared between " << parameter->ToString() << " and "
                       << GetPrimName(cnode);
        } else {
          MS_LOG(ERROR) << "Can not find the shared AllGather with multiple node users.";
        }
      } else {
        // insert allgather operator between shard parameter and cnode
        auto param_ptr = parameter->cast<ParameterPtr>();
        MS_EXCEPTION_IF_NULL(param_ptr);
        bool is_shared_param = param_ptr->user_data<TensorLayout>()->is_shared_param();
        InsertAllGatherOp(root, opt_shard_group, param_pair, parameter, op_name, is_shared_param);
        insert_flag = true;
      }
    }
  }
}

void SetSharedParameterFlag(const FuncGraphPtr &root, const AnfNodePtr &parameter) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(parameter);
  FuncGraphManagerPtr manager = root->manager();
  MS_EXCEPTION_IF_NULL(manager);
  ParameterPtr parameter_ptr = parameter->cast<ParameterPtr>();
  if (parameter_ptr == nullptr) {
    MS_LOG(INFO) << parameter->ToString() << ": cast to ptr failed. it may not be a parameter";
    return;
  }
  auto user_set = manager->node_users()[parameter];
  int32_t user_count = 0;
  for (auto &param_pair : user_set) {
    CNodePtr cnode = param_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->in_forward_flag()) user_count++;
  }
  if (user_count > 1) {
    auto tensor_layout = parameter_ptr->user_data<TensorLayout>();
    tensor_layout->set_is_shared_param(true);
    MS_LOG(WARNING) << "There are multiple users for " << parameter->ToString()
                    << ". Mixed precision optimization is not valid here.";
  }
}

// When this function returns non-empty string, that means parallel optimizer is applied on this parameter.
std::string SetParallelShape(const AnfNodePtr &parameter, const std::pair<AnfNodePtr, int64_t> &res,
                             const FuncGraphPtr &root) {
  // check null for param and cnode
  auto param_shape = parameter->Shape();

  MS_EXCEPTION_IF_NULL(parameter);
  MS_EXCEPTION_IF_NULL(param_shape);

  CNodePtr cnode = res.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  // get slice_shape
  OperatorInfoPtr distribute_operator = cnode->user_data<OperatorInfo>();
  if (distribute_operator == nullptr) {
    MS_LOG(EXCEPTION) << "node " << cnode->ToString() << " 's distribute_operator is nullptr";
  }
  if (LongToSize(res.second - 1) >= distribute_operator->inputs_tensor_info().size()) {
    MS_LOG(EXCEPTION) << "The parameter index is not in inputs_tensor_info. index = " << (res.second - 1)
                      << ", inputs_tensor_info size = " << distribute_operator->inputs_tensor_info().size();
  }
  TensorInfo tensorinfo_in = distribute_operator->inputs_tensor_info()[LongToSize(res.second - 1)];
  TensorLayout tensor_layout = tensorinfo_in.tensor_layout();
  Shape slice_shape = tensor_layout.slice_shape().array();

  // generate shard group
  std::string opt_shard_group;
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  bool enable_parallel_optimizer = ParallelContext::GetInstance()->enable_parallel_optimizer();
  if (enable_parallel_optimizer) {
    std::unique_ptr<OptParamMgr> apOptParamMgr = createOptParamMgr(root);
    opt_shard_group = apOptParamMgr->ShardOptGroup(parameter, &tensor_layout, distribute_operator);
    // set the shape of parameter to sliced shape
    if (!opt_shard_group.empty()) {
      slice_shape = tensor_layout.opt_shard_slice_shape();
    }
    MS_LOG(INFO) << "the shape of " << parameter->ToString() << "(original: " << param_shape->ToString() << ")"
                 << " will be sliced into " << MakeValue(slice_shape)->ToString() << " in op "
                 << distribute_operator->name();
  }

  AbstractBasePtr abstract = parameter->abstract();
  if (abstract == nullptr) {
    MS_LOG(EXCEPTION) << "parameter " << parameter->ToString() << ": abstract is nullptr";
  }

  AbstractBasePtr cloned_abstract = abstract->Clone();
  if (cloned_abstract == nullptr) {
    MS_LOG(EXCEPTION) << "parameter " << parameter->ToString() << ": abstract clone failed";
  }

  cloned_abstract->set_shape(std::make_shared<abstract::Shape>(slice_shape));
  parameter->set_abstract(cloned_abstract);
  ParameterPtr parameter_ptr = parameter->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(parameter_ptr);
  parameter_ptr->set_user_data<TensorLayout>(std::make_shared<TensorLayout>(tensor_layout));
  return opt_shard_group;
}

void CoverSliceShape(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  auto parameters = root->parameters();
  for (auto &parameter : parameters) {
    MS_EXCEPTION_IF_NULL(parameter->Shape());

    auto iter = g_RefMap.find(parameter);
    if (iter != g_RefMap.end()) {
      std::string group = SetParallelShape(parameter, g_RefMap[parameter], root);
      // find all forward nodes that use parameter in graphs and insert allgather if group is not empty
      SetSharedParameterFlag(root, parameter);
      ApplyParallelOptOnParam(root, parameter, group);
      continue;
    }

    std::pair<AnfNodePtr, int64_t> res = FindSubGraph(root, parameter);
    if (res.first == nullptr) {
      MS_LOG(INFO) << "Parameter " << parameter->ToString() << " is not in graph, thus no need to set parallel shape";
    } else {
      std::string group = SetParallelShape(parameter, res, root);
      // find all forward nodes that use parameter in graphs and insert allgather if group is not empty
      SetSharedParameterFlag(root, parameter);
      ApplyParallelOptOnParam(root, parameter, group);
      MS_LOG(DEBUG) << "Parameter " << parameter->ToString() << " shape " << parameter->Shape()->ToString();
    }
  }
  g_RefMap.clear();
}

void SetVirtualDatasetStrategy(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  bool full_batch = ParallelContext::GetInstance()->full_batch();

  PrimitivePtr prim = GetValueNode<PrimitivePtr>(node->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->name() == VIRTUAL_DATA_SET || prim->name() == VIRTUAL_OUTPUT) {
    CheckGlobalDeviceManager();
    auto attrs_temp = prim->attrs();
    if (!ParallelContext::GetInstance()->dataset_strategy().empty() && prim->name() == VIRTUAL_DATA_SET) {
      std::vector<ValuePtr> elements;
      auto dataset_strategy = ParallelContext::GetInstance()->dataset_strategy();
      (void)std::transform(dataset_strategy.begin(), dataset_strategy.end(), std::back_inserter(elements),
                           [](auto input_stra) { return MakeValue(input_stra); });
      ValueTuplePtr strategy = std::make_shared<ValueTuple>(elements);
      attrs_temp[IN_STRATEGY] = strategy;
      (void)prim->SetAttrs(attrs_temp);
      if (prim->HasAttr(REPEAT_DIM_DIRECT) && GetValue<std::string>(prim->GetAttr(REPEAT_DIM_DIRECT)) == RIGHT) {
        ParallelContext::GetInstance()->set_dataset_repeat_dim_right(true);
        MS_LOG(INFO) << "dataset repeat dim is right";
      }
      return;
    }
    int64_t dev_num;
    if (full_batch) {
      dev_num = 1;
    } else {
      dev_num = g_device_manager->stage_device_num();
    }
    if (dev_num == 0) {
      MS_LOG(EXCEPTION) << "Device Num must be larger than 0, but got 0.";
    }
    std::vector<Shapes> shape_list = ExtractShape(node);
    if (shape_list.empty()) {
      MS_LOG(EXCEPTION) << "Failure:node " << node->ToString() << " failed to extract shape";
    }
    std::vector<ValuePtr> elements;
    for (size_t i = 0; i < shape_list[0].size(); i++) {
      if (shape_list[0][i].empty()) {
        MS_LOG(EXCEPTION) << "shape_list[ " << i << " ].size() is zero";
      }
      Dimensions input_strategy;
      if (!shape_list[0][i].empty() && shape_list[0][i][0] % dev_num == 0) {
        input_strategy.push_back(dev_num);
      } else if (!shape_list[0][i].empty()) {
        input_strategy.push_back(1);
      }
      for (size_t j = 1; j < shape_list[0][i].size(); j++) {
        input_strategy.push_back(1);
      }
      elements.push_back(MakeValue(input_strategy));
    }
    ValueTuplePtr strategy = std::make_shared<ValueTuple>(elements);
    attrs_temp[IN_STRATEGY] = strategy;
    (void)prim->SetAttrs(attrs_temp);
  }
}

// find previous parallel care node's next node.
bool FindPreNodes(const AnfNodePtr &node, std::vector<std::string> *unique_ids, std::vector<size_t> *indexes,
                  size_t curr_depth) {
  if (curr_depth > MAX_RECURSIVE_DEPTH) {
    MS_LOG(WARNING) << "When find the previous node, exceeded the maximum recursion depth: " << MAX_RECURSIVE_DEPTH;
    return false;
  }
  MS_EXCEPTION_IF_NULL(unique_ids);
  MS_EXCEPTION_IF_NULL(indexes);
  if (!node->isa<CNode>()) {
    return false;
  }
  CNodePtr pre_cnode = node->cast<CNodePtr>();
  if (!IsValueNode<Primitive>(pre_cnode->input(0))) {
    return false;
  }
  bool find = false;
  for (size_t index = 1; index < pre_cnode->inputs().size(); ++index) {
    auto next_node = pre_cnode->inputs()[index];
    if (!next_node->isa<CNode>() || next_node->isa<Parameter>()) {
      return false;
    }
    CNodePtr cnode = next_node->cast<CNodePtr>();
    if (!IsValueNode<Primitive>(cnode->input(0))) {
      return false;
    }
    ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
    PrimitivePtr prim = prim_anf_node->value()->cast<PrimitivePtr>();
    if (IsParallelCareNode(cnode) && prim->name() != MAKE_TUPLE && prim->name() != MAKE_LIST) {
      unique_ids->push_back(pre_cnode->UniqueId());
      indexes->push_back(index);
      find = true;
      continue;
    }
    if (FindPreNodes(cnode, unique_ids, indexes, ++curr_depth)) {
      find = true;
      continue;
    }
  }
  return find;
}

void FindLastNodesUniqueId(const FuncGraphPtr &root, std::vector<std::string> *unique_ids,
                           std::vector<size_t> *indexes) {
  MS_EXCEPTION_IF_NULL(unique_ids);
  CNodePtr cnode = root->get_return();
  if (!FindPreNodes(cnode, unique_ids, indexes, 0)) {
    MS_LOG(WARNING) << "cannot find the last parallel care node in eval graph";
  }
}

StrategyPtr GenerateBatchParallelStrategy(const OperatorInfoPtr operator_, const PrimitivePtr prim) {
  MS_EXCEPTION_IF_NULL(operator_);
  MS_EXCEPTION_IF_NULL(prim);
  StrategyPtr strategyPtr;
  std::shared_ptr<Strategys> strategy_v_ptr = operator_->GenerateBatchStrategies();
  MS_EXCEPTION_IF_NULL(strategy_v_ptr);
  strategyPtr = NewStrategy(0, *strategy_v_ptr);
  std::vector<ValuePtr> elements;
  for (size_t i = 0; i < strategy_v_ptr->size(); i++) {
    elements.push_back(MakeValue((*strategy_v_ptr)[i]));
  }
  ValueTuplePtr strategy = std::make_shared<ValueTuple>(elements);
  // display the strategy generated by batch parallel
  auto attrs = prim->attrs();
  attrs[GEN_STRATEGY] = strategy;
  (void)prim->SetAttrs(attrs);
  MS_LOG(INFO) << "prim " << prim->name() << " batch parallel strategy is " << attrs[GEN_STRATEGY]->ToString();
  return strategyPtr;
}

static bool CheckExtractInfomation(const CNodePtr &cnode) {
  if ((cnode == nullptr) || !IsValueNode<Primitive>(cnode->input(0))) {
    return false;
  }

  ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);
  if ((prim->name() == MAKE_TUPLE) || (prim->name() == MAKE_LIST) || (prim->name() == RECEIVE)) {
    return false;
  }

  if (!IsParallelCareNode(cnode)) {
    return false;
  }
  return true;
}

static void ExtractStrategyAndInit(const CNodePtr &cnode, const PrimitivePtr &prim, const OperatorInfoPtr &op_info) {
  StrategyPtr in_strategy = nullptr, out_strategy = nullptr;
  auto attrs = prim->attrs();

  // load strategy map from checkpoint
  StrategyMap stra_map;
  if (StrategyCheckpoint::GetInstance().LoadCheckPointOn() &&
      (StrategyCheckpoint::GetInstance().Load(&stra_map) != SUCCESS)) {
    MS_LOG(EXCEPTION) << "Load strategy checkpoint failed";
  }

  std::string strategy_key_name = "";
  auto param_names = NodeParameterName(cnode, -1, 0);
  if (!param_names.empty()) {
    strategy_key_name = prim->name() + "_" + param_names[0].first;
  }
  bool load_strategy_from_ckpt =
    StrategyCheckpoint::GetInstance().LoadCheckPointOn() && stra_map.find(strategy_key_name) != stra_map.end();
  if ((!StrategyFound(attrs) && !load_strategy_from_ckpt) && !cnode->HasPrimalAttr(IN_STRATEGY)) {
    MS_LOG(INFO) << "ExtractInformation: the strategy of node " << cnode->ToString() << " prim " << prim->name()
                 << " is empty, using batch parallel";
    in_strategy = GenerateBatchParallelStrategy(op_info, prim);
  } else if (cnode->HasPrimalAttr(IN_STRATEGY)) {
    in_strategy = ExtractStrategy(cnode->GetPrimalAttr(IN_STRATEGY));
    out_strategy = ExtractStrategy(cnode->GetPrimalAttr(OUT_STRATEGY));
  } else if (StrategyFound(attrs)) {
    in_strategy = ExtractStrategy(attrs[IN_STRATEGY]);
    out_strategy = ExtractStrategy(attrs[OUT_STRATEGY]);
  } else {
    in_strategy = stra_map[strategy_key_name];
  }

  MS_EXCEPTION_IF_NULL(in_strategy);
  if (op_info->Init(in_strategy, out_strategy) == FAILED) {
    MS_LOG(EXCEPTION) << "Failure:operator " << prim->name() << " init failed" << trace::DumpSourceLines(cnode);
  }
}

void ExtractInformation(const std::vector<AnfNodePtr> &all_nodes) {
  SetStridedSliceSplitStrategy(all_nodes);
  for (auto &node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (!CheckExtractInfomation(cnode) || IsPrimitiveCNode(node, prim::kPrimSend)) {
      continue;
    }

    SetVirtualDatasetStrategy(cnode);
    ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);

    auto attrs = prim->attrs();
    MS_LOG(INFO) << "extract information: node: " << node->ToString() << " prim " << prim->name();

    std::vector<Shapes> shape_list = ExtractShape(cnode);
    if (shape_list.empty()) {
      MS_LOG(EXCEPTION) << "Failure:node " << node->ToString() << " failed to extract shape";
    }
    OperatorInfoPtr operator_ = OperatorInstance(prim, attrs, shape_list);
    MS_EXCEPTION_IF_NULL(operator_);

    auto &inputs = cnode->inputs();
    std::vector<ValuePtr> input_value;
    for (size_t index = 1; index < inputs.size(); ++index) {
      if (inputs[index]->isa<ValueNode>()) {
        input_value.push_back(GetValueNode(inputs[index]));
        continue;
      }
      input_value.emplace_back(nullptr);
    }

    (*operator_).set_input_value(input_value);
    (*operator_).set_outputs_dtype(cnode->Type());
    (*operator_).set_cnode(cnode);
    if (prim->name() == RESHAPE) {
      cnode->set_user_data<OperatorInfo>(operator_);
      continue;
    }

    ExtractStrategyAndInit(cnode, prim, operator_);
    cnode->set_user_data<OperatorInfo>(operator_);
  }
}

TensorLayout GetInputLayoutFromCNode(const std::pair<AnfNodePtr, int64_t> &node_pair) {
  CNodePtr cnode = node_pair.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  OperatorInfoPtr distribute_operator = GetDistributeOperator(cnode);
  MS_EXCEPTION_IF_NULL(distribute_operator);
  int64_t index = node_pair.second;
  if (index > SizeToLong(distribute_operator->inputs_tensor_info().size())) {
    MS_LOG(EXCEPTION) << "The index is out of range, the node_pair.second is  " << (index - 1)
                      << ", the vector size is  " << distribute_operator->inputs_tensor_info().size();
  }
  TensorInfo tensorinfo_in = distribute_operator->inputs_tensor_info()[LongToSize(index - 1)];
  TensorLayout tensorlayout_in = tensorinfo_in.tensor_layout();
  return tensorlayout_in;
}

// if reshape's output connect to several primitive, return the first layout found
std::shared_ptr<TensorLayout> FindNextLayout(const CNodePtr &cnode, bool *next_is_reshape) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(cnode->func_graph());
  FuncGraphManagerPtr manager = cnode->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodeIndexSet node_set = manager->node_users()[cnode];
  for (auto &node_pair : node_set) {
    CNodePtr use_apply = node_pair.first->cast<CNodePtr>();
    if (use_apply == nullptr || !IsValueNode<Primitive>(use_apply->input(0))) {
      continue;
    }
    if (IsPrimitiveCNode(use_apply, prim::kPrimReshape)) {
      *next_is_reshape = true;
      continue;
    }
    ValueNodePtr prim_anf_node = use_apply->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prim_anf_node);
    PrimitivePtr node_prim = prim_anf_node->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(node_prim);
    MS_LOG(INFO) << "FindNextLayout prim " << node_prim->name();
    if (node_prim->name() == DEPEND && node_pair.second != 1) {
      continue;
    }
    if (IsParallelCareNode(use_apply) && use_apply->has_user_data<OperatorInfo>()) {
      MS_LOG(INFO) << "FindNextLayout success prim " << node_prim->name();
      *next_is_reshape = false;
      auto layout = GetInputLayoutFromCNode(node_pair);
      return std::make_shared<TensorLayout>(layout);
    }
    MS_LOG(DEBUG) << "FindNextLayout failed prim " << node_prim->name() << "  " << IsParallelCareNode(use_apply)
                  << "   " << use_apply->has_user_data<OperatorInfo>();

    auto layout_ptr = FindNextLayout(use_apply, next_is_reshape);
    if (layout_ptr) {
      return layout_ptr;
    }
  }
  MS_LOG(WARNING) << "FindNextLayout return nullptr, if reshape is not the last primitive, there must be some error";
  return nullptr;
}

std::shared_ptr<TensorLayout> GetOutputLayoutFromCNode(const CNodePtr &cnode, size_t output_index) {
  MS_EXCEPTION_IF_NULL(cnode);
  OperatorInfoPtr distribute_operator = GetDistributeOperator(cnode);
  MS_EXCEPTION_IF_NULL(distribute_operator);
  if (distribute_operator->outputs_tensor_info().size() <= output_index) {
    MS_LOG(EXCEPTION) << "outputs_tensor_info size is  " << distribute_operator->inputs_tensor_info().size()
                      << ", must be greater than output_index  " << output_index;
  }
  TensorInfo tensorinfo_out = distribute_operator->outputs_tensor_info()[output_index];
  TensorLayout tensorlayout_out = tensorinfo_out.tensor_layout();
  return std::make_shared<TensorLayout>(tensorlayout_out);
}

std::shared_ptr<TensorLayout> FindPrevParallelCareNodeLayout(const AnfNodePtr &node, size_t output_index) {
  if (!node->isa<CNode>()) {
    return nullptr;
  }
  CNodePtr cnode = node->cast<CNodePtr>();
  if (!IsValueNode<Primitive>(cnode->input(0))) {
    return nullptr;
  }
  if (IsParallelCareNode(cnode) && cnode->has_user_data<OperatorInfo>()) {
    auto layout_ptr = GetOutputLayoutFromCNode(cnode, output_index);
    if (!layout_ptr) {
      MS_LOG(EXCEPTION) << "Failure:GetLayoutFromCNode failed";
    }
    return layout_ptr;
  }
  return nullptr;
}

std::shared_ptr<TensorLayout> FindParameterNextLayout(const AnfNodePtr &node, size_t curr_depth) {
  if (curr_depth > MAX_RECURSIVE_DEPTH) {
    MS_LOG(WARNING) << "When finding the next tensor layout for the parameter, exceeded the maximum recursion depth: "
                    << MAX_RECURSIVE_DEPTH;
    return nullptr;
  }
  FuncGraphManagerPtr manager = node->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodeIndexSet node_set = manager->node_users()[node];
  for (auto &node_pair : node_set) {
    if (IsPrimitiveCNode(node_pair.first, prim::kPrimLoad)) {
      auto layout_param = FindParameterNextLayout(node_pair.first, ++curr_depth);
      if (!layout_param) {
        continue;
      }
      return layout_param;
    }
    CNodePtr use_apply = node_pair.first->cast<CNodePtr>();
    if (use_apply == nullptr || !IsValueNode<Primitive>(use_apply->input(0))) {
      continue;
    }
    ValueNodePtr prim_anf_node = use_apply->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prim_anf_node);
    PrimitivePtr node_prim = prim_anf_node->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(node_prim);
    if ((node_prim->name() == DEPEND && node_pair.second != 1) || node_prim->name() == RESHAPE) {
      continue;
    }
    if (IsParallelCareNode(use_apply) && use_apply->has_user_data<OperatorInfo>()) {
      auto layout = GetInputLayoutFromCNode(node_pair);
      return std::make_shared<TensorLayout>(layout);
    }
  }
  return nullptr;
}

std::shared_ptr<TensorLayout> CreateParameterLayout(const AnfNodePtr &node) {
  // Create DataParallel tensor layout for parameter(support WideDeep).
  auto next_layout = FindParameterNextLayout(node, 0);
  if (next_layout != nullptr) {
    return next_layout;
  }
  CheckGlobalDeviceManager();
  int64_t dev_num = g_device_manager->stage_device_num();
  TensorLayout input_tensor_layout;
  // create input_shape
  Shapes inputs_shape = GetNodeShape(node);
  Shape input_shape_array = inputs_shape[0];
  if (input_shape_array.empty()) {
    MS_LOG(EXCEPTION) << "Don't support reshape a scalar parameter.";
  }
  // create tensor_map
  size_t shape_size = input_shape_array.size();
  TensorMap input_tensor_map_array(SizeToLong(shape_size) - 1, -1);
  input_tensor_map_array.insert(input_tensor_map_array.begin(), 0);
  // create dev_matrix
  Shape dev_matrix_array = {dev_num};
  if (input_tensor_layout.InitFromVector(dev_matrix_array, input_tensor_map_array, input_shape_array) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Create tensor layout for parameter failed.";
  }
  return std::make_shared<TensorLayout>(input_tensor_layout);
}

RedistributionOpListPtr InferSensRedistribution(const AnfNodePtr &node, const TensorLayout &loss_layout) {
  MS_EXCEPTION_IF_NULL(node);
  TensorRedistribution tensor_redistribution;
  // create stand alone layout:TensorMap:[all -1],dev_matrix:[dev_num].
  CheckGlobalDeviceManager();
  int64_t dev_num = g_device_manager->stage_device_num();
  TensorLayout stand_alone_layout;
  Shapes inputs_shape = GetNodeShape(node);
  if (inputs_shape.empty()) {
    MS_LOG(EXCEPTION) << "InferSensRedistribution failed cause inputs shape is empty.";
  }
  Shape input_shape_array = inputs_shape[0];
  if (input_shape_array.empty()) {
    MS_LOG(INFO) << "No need to redistribution for sens.";
    return nullptr;
  }
  // TensorMap
  TensorMap stand_alone_tensor_map_array(SizeToLong(input_shape_array.size()), -1);
  // Dev_matrix
  Shape dev_matrix_array = {dev_num};
  if (stand_alone_layout.InitFromVector(dev_matrix_array, stand_alone_tensor_map_array, input_shape_array) == FAILED) {
    MS_LOG(EXCEPTION) << "Create tensor layout for Sens failed.";
  }

  // Infer Redistribution op list for stand alone and loss layout.
  RankList dev_list = g_device_manager->GetDeviceListInThisStage();
  if (tensor_redistribution.Init(stand_alone_layout, loss_layout, dev_list) == FAILED) {
    MS_LOG(EXCEPTION) << "Redistribution for Sens init failed.";
  }
  RedistributionOpListPtr sens_redistribution_list = tensor_redistribution.InferTensorRedistributionOperatorList();
  MS_EXCEPTION_IF_NULL(sens_redistribution_list);

  return sens_redistribution_list;
}

std::shared_ptr<TensorLayout> FindPrevLayout(const AnfNodePtr &node) {
  if (node->isa<Parameter>()) {
    return CreateParameterLayout(node);
  }
  if (!node->isa<CNode>()) {
    return nullptr;
  }
  CNodePtr cnode = node->cast<CNodePtr>();
  if (!IsValueNode<Primitive>(cnode->input(0))) {
    return nullptr;
  }
  if (IsPrimitiveCNode(node, prim::kPrimReceive)) {
    return cnode->user_data<TensorLayout>();
  }
  if (IsParallelCareNode(cnode) && cnode->has_user_data<OperatorInfo>() &&
      !IsPrimitiveCNode(node, prim::kPrimReshape)) {
    auto layout_ptr = GetOutputLayoutFromCNode(cnode, 0);
    if (!layout_ptr) {
      MS_LOG(EXCEPTION) << "Failure:GetLayoutFromCNode failed";
    }
    return layout_ptr;
  }
  ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
  PrimitivePtr prim = prim_anf_node->value()->cast<PrimitivePtr>();
  if (prim->name() == prim::kTupleGetItem) {
    auto tuple_index = GetTupleGetItemIndex(cnode);
    auto layout_ptr = FindPrevParallelCareNodeLayout(cnode->input(1), LongToSize(tuple_index));
    if (!layout_ptr) {
      MS_LOG(EXCEPTION) << " Failure:FindPrevLayout failed, tuple_getitem before reshape, but there does not exit a "
                           "parallel care node "
                           "before tuple_getitem!";
    }
    return layout_ptr;
  }
  for (size_t index = 0; index < cnode->inputs().size(); ++index) {
    if (prim->name() == DEPEND && index != 1) {
      continue;
    }
    auto layout_ptr = FindPrevLayout(cnode->inputs()[index]);
    if (!layout_ptr) {
      continue;
    }
    return layout_ptr;
  }
  MS_LOG(WARNING) << "FindPrevLayout return nullptr, if reshape is not the first primitive, there must be some error";
  return nullptr;
}

void ReshapeInit(const std::vector<AnfNodePtr> &all_nodes) {
  for (auto &node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if ((cnode == nullptr) || !IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
    if (!IsParallelCareNode(cnode) || !cnode->has_user_data<OperatorInfo>()) {
      continue;
    }
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);
    MS_EXCEPTION_IF_NULL(prim);
    OperatorInfoPtr operator_info = cnode->user_data<OperatorInfo>();
    if (operator_info == nullptr) {
      MS_LOG(EXCEPTION) << "Failure:Primitive " << prim->ToString() << " OperatorInstance is nullptr";
    }
    if (prim->name() != RESHAPE) {
      continue;
    }
    auto attrs = prim->attrs();
    if (StrategyFound(attrs)) {
      MS_LOG(EXCEPTION) << "Setting strategy for Reshape goes for nothing!";
    }
    MS_ASSERT(cnode->inputs().size() == RESHAPE_INPUT_SIZE);
    auto prev_layout_ptr = FindPrevLayout(cnode->input(1));
    if (prev_layout_ptr) {
      auto reshape_info_ptr = std::dynamic_pointer_cast<ReshapeInfo>(operator_info);
      reshape_info_ptr->SetInputLayout(*prev_layout_ptr);
    }
    bool is_next_reshape = false;
    auto next_layout_ptr = FindNextLayout(cnode, &is_next_reshape);
    if (next_layout_ptr) {
      auto reshape_info_ptr = std::dynamic_pointer_cast<ReshapeInfo>(operator_info);
      reshape_info_ptr->SetOutputLayout(*next_layout_ptr);
    } else if (is_next_reshape && prev_layout_ptr != nullptr) {
      auto reshape_info_ptr = std::dynamic_pointer_cast<ReshapeInfo>(operator_info);
      reshape_info_ptr->SetOutputLayout(*prev_layout_ptr);
    }
    if (operator_info->Init(nullptr, nullptr) == FAILED) {
      MS_LOG(EXCEPTION) << "Failure:operator " << prim->ToString() << " init failed";
    }
  }
}

CNodePtr HandleDependLoss(const CNodePtr &cnode, size_t curr_depth) {
  if (curr_depth > MAX_RECURSIVE_DEPTH) {
    MS_LOG(WARNING) << "When handling the loss node of Depend, exceeded the max recursive depth: "
                    << MAX_RECURSIVE_DEPTH;
    return nullptr;
  }
  // Handle return->depend->loss
  if (IsPrimitiveCNode(cnode, prim::kPrimDepend) ||
      (IsPrimitiveCNode(cnode, prim::kPrimCast) && !cnode->has_user_data<OperatorInfo>())) {
    auto depend_before = cnode->input(1)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(depend_before);
    return HandleDependLoss(depend_before, ++curr_depth);
  }
  return cnode;
}

LossNodeInfo FindLossCNode(const FuncGraphPtr &func_graph, size_t max_depth) {
  if (max_depth > MAX_RECURSIVE_DEPTH) {
    MS_LOG(EXCEPTION) << "Recursive call is larger than 100000.";
  }
  LossNodeInfo loss_node_info;
  MS_EXCEPTION_IF_NULL(func_graph);
  CNodePtr return_node = func_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  if (return_node->size() < 2) {
    MS_LOG(EXCEPTION) << "Failure: " << return_node->DebugString() << " size is smaller than 2";
  }
  AnfNodePtr pre_node = return_node->input(1);
  MS_EXCEPTION_IF_NULL(pre_node);
  auto pre_cnode = pre_node->cast<CNodePtr>();
  pre_cnode = HandleDependLoss(pre_cnode, 0);
  if (pre_cnode->input(0)->isa<CNode>()) {
    auto switch_cnode = pre_cnode->input(0)->cast<CNodePtr>();
    if (IsPrimitiveCNode(switch_cnode, prim::kPrimSwitch)) {
      MS_EXCEPTION_IF_NULL(switch_cnode);
      auto switch_graph = GetValueNode<FuncGraphPtr>(switch_cnode->input(2));
      return FindLossCNode(switch_graph, max_depth + 1);
    }
  }
  if (pre_cnode == nullptr || !IsValueNode<Primitive>(pre_cnode->input(0))) {
    return loss_node_info;
  }
  if (!IsValueNode<Primitive>(pre_cnode->input(0))) {
    MS_LOG(DEBUG) << "pre_cnode:" << pre_cnode->ToString();
    return loss_node_info;
  }
  auto current_prim = GetValueNode<PrimitivePtr>(pre_cnode->input(0));
  // notice: the GetNext op has not input
  if (INVALID_LOSS_OPS.find(current_prim->name()) != INVALID_LOSS_OPS.end()) {
    MS_LOG(INFO) << "The loss is: " << current_prim->name();
    loss_node_info.loss_node = pre_cnode;
    return loss_node_info;
  }

  // size of common cnode is larger than 1
  if (pre_cnode->size() < 2) {
    MS_LOG(EXCEPTION) << pre_cnode->ToString() << " size( " << pre_cnode->inputs().size() << " ) is smaller than 2";
  }

  // return -> tuple_getitem -> loss
  if (current_prim->name() == prim::kTupleGetItem) {
    auto tuple_index = GetTupleGetItemIndex(pre_cnode);
    AnfNodePtr pre_pre_node = pre_cnode->input(1);
    MS_EXCEPTION_IF_NULL(pre_pre_node);

    auto pre_pre_cnode = pre_pre_node->cast<CNodePtr>();
    loss_node_info.has_tuple_getitem = true;
    loss_node_info.dout_index = tuple_index;
    loss_node_info.loss_node = pre_pre_cnode;
    return loss_node_info;
  }

  // return -> make_tuple
  if (current_prim->name() == MAKE_TUPLE) {
    MS_LOG(WARNING) << "The loss have make_tuple, it is not supported";
    return loss_node_info;
  }

  // return -> loss
  loss_node_info.loss_node = pre_cnode;
  MS_LOG(DEBUG) << "The loss name is " << current_prim->name();
  return loss_node_info;
}

TensorLayouts GetLossNodeGradOutputLayout(const LossNodeInfo &node_info) {
  TensorLayouts ret;
  auto loss_cnode = node_info.loss_node;
  MS_EXCEPTION_IF_NULL(loss_cnode);

  ValueNodePtr prim_anf_node = loss_cnode->input(0)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(prim_anf_node);
  PrimitivePtr prim = prim_anf_node->value()->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(prim);
  if (INVALID_LOSS_OPS.find(prim->name()) != INVALID_LOSS_OPS.end()) {
    MS_LOG(WARNING) << "The loss name is: " << prim->name() << ", do nothing for split sens now";
    return ret;
  }

  OperatorInfoPtr operator_info = loss_cnode->user_data<OperatorInfo>();
  MS_EXCEPTION_IF_NULL(operator_info);
  TensorInfo loss_grad_tensor_info;
  size_t op_output_size = operator_info->outputs_tensor_info().size();
  MS_LOG(INFO) << "The loss name is " << operator_info->name() << ", the has tuple item is  "
               << node_info.has_tuple_getitem << ", the output size is  " << op_output_size << ", the dout_index is  "
               << node_info.dout_index;

  if ((op_output_size == 0) || (op_output_size <= LongToSize(node_info.dout_index))) {
    MS_LOG(EXCEPTION) << "The index is  " << node_info.dout_index << ", but the size of outputs is  " << op_output_size;
  }

  if (!node_info.has_tuple_getitem && (op_output_size > 1)) {
    MS_LOG(EXCEPTION) << "Currently, it is not supported that the sens is a tuple.";
  }

  loss_grad_tensor_info = operator_info->outputs_tensor_info()[LongToSize(node_info.dout_index)];
  ret.push_back(loss_grad_tensor_info.tensor_layout());
  return ret;
}

void SplitSens(const CNodePtr &grad_sens_node, const TensorLayout &loss_grad_layout) {
  MS_EXCEPTION_IF_NULL(grad_sens_node);
  if (grad_sens_node->size() <= 1) {
    MS_LOG(EXCEPTION) << "The size of grad sens node is smaller than 2";
  }
  AnfNodePtr sens_tensor_node = grad_sens_node->input(1);
  MS_EXCEPTION_IF_NULL(sens_tensor_node);
  Shapes sens_shapes = GetNodeShape(sens_tensor_node);
  if (sens_shapes.size() != 1) {
    MS_LOG(EXCEPTION) << "GetNodeShape for sens_tensor_node, output size is not 1";
  }
  // If the shape of sens tensor is [] or [1], no need to split it.
  Shape sens_shape = sens_shapes[0];
  if (sens_shape.empty() || ((sens_shape.size() == 1) && (sens_shape[0] == 1))) {
    if (sens_tensor_node->isa<Parameter>()) {
      auto sens_tensor_param = sens_tensor_node->cast<ParameterPtr>();
      MS_LOG(DEBUG) << "loss layout " << loss_grad_layout.ToString();
      sens_tensor_param->set_user_data<TensorLayout>(std::make_shared<TensorLayout>(loss_grad_layout));
    }
    MS_LOG(INFO) << "The shape of sens is " << ShapeToString(sens_shape) << ", no need to split sens";
    return;
  }
  auto loss_shape = loss_grad_layout.tensor_shape().array();
  if (loss_shape != sens_shape) {
    MS_LOG(EXCEPTION) << "The shape of sens is not equal to loss output, it is unsupported now. Sens shape is "
                      << ShapeToString(sens_shape) << ", loss shape is " << ShapeToString(loss_shape);
  }
  MS_LOG(INFO) << "The shape of sens is " << ShapeToString(sens_shape) << ", split it.";

  if (!IsValueNode<Tensor>(sens_tensor_node)) {
    if (sens_tensor_node->isa<Parameter>()) {
      MS_LOG(DEBUG) << "loss layout " << loss_grad_layout.ToString();
      AbstractBasePtr abstract = sens_tensor_node->abstract();
      MS_EXCEPTION_IF_NULL(abstract);
      auto slice_shape = loss_grad_layout.slice_shape().array();
      std::shared_ptr<abstract::BaseShape> parallel_shape = std::make_shared<abstract::Shape>(slice_shape);
      MS_EXCEPTION_IF_NULL(parallel_shape);
      auto cloned_abstract = abstract->Clone();
      MS_EXCEPTION_IF_NULL(cloned_abstract);
      cloned_abstract->set_shape(parallel_shape);
      sens_tensor_node->set_abstract(cloned_abstract);
      auto sens_tensor_param = sens_tensor_node->cast<ParameterPtr>();
      sens_tensor_param->set_user_data<TensorLayout>(std::make_shared<TensorLayout>(loss_grad_layout));
      return;
    }
    if (sens_tensor_node->isa<CNode>()) {
      auto op_list_ptr = InferSensRedistribution(sens_tensor_node, loss_grad_layout);
      if (op_list_ptr == nullptr) {
        return;
      }
      auto sens_tensor_cnode = sens_tensor_node->cast<CNodePtr>();
      auto func_graph = grad_sens_node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      InsertRedistribution(op_list_ptr, grad_sens_node, func_graph, 1, sens_tensor_cnode);
      return;
    }
    MS_LOG(EXCEPTION) << "The type of sens node is not Tensor or Parameter or CNode, it is unsupported now.";
  }

  // Use _GetTensorSlice operator to split the sens tensor
  FuncGraphPtr func_graph = grad_sens_node->func_graph();  // only cnode can get the graph
  MS_EXCEPTION_IF_NULL(func_graph);
  Operator op = CreateGetTensorSliceOp(loss_grad_layout);
  InsertGetTensorSliceOp(op, grad_sens_node, func_graph, 1, SPLIT_SENS);
}

void InsertForwardOps(const OperatorInfoPtr &distribute_operator, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(distribute_operator);
  MS_EXCEPTION_IF_NULL(cnode);
  if (IsPrimitiveCNode(cnode, prim::kPrimReceive)) {
    return;
  }
  OperatorVector forward_op = distribute_operator->forward_op();
  if (!forward_op.empty()) {
    MS_LOG(INFO) << "Insert forward op for " << distribute_operator->name();
    ForwardCommunication(forward_op, cnode);
  }
}

void StepReplace(const OperatorInfoPtr &distribute_operator, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(distribute_operator);
  MS_EXCEPTION_IF_NULL(cnode);
  // StepReplaceOp
  OperatorVector replace_op = distribute_operator->replace_op();
  if (!replace_op.empty()) {
    MS_LOG(INFO) << "StepReplaceOp " << cnode->ToString();
    StepReplaceOp(replace_op, cnode);
  }

  // StepReplaceGraph: after calling StepReplaceGraph, cnode can not be used anymore.
  ReplaceGraphPtr replace_graph = distribute_operator->replace_graph(cnode);
  if (!replace_op.empty() && replace_graph) {
    MS_LOG(EXCEPTION) << "Only one of replace_op or replace_op can be used";
  }
  if (replace_graph) {
    MS_LOG(INFO) << "StepReplaceGraph " << cnode->ToString();
    StepReplaceGraph(replace_graph, cnode);
  }
}

std::set<FuncGraphPtr> FindForwardGraphByRootNodes(const AnfNodeSet &root_all_nodes) {
  // J->CNode->Graph
  std::set<FuncGraphPtr> graph_set;
  for (auto &node : root_all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }

    auto cnode = node->cast<CNodePtr>();
    if ((cnode->size() < 2) || !IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    auto expect_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (expect_prim->name() != J && expect_prim->name() != SHARD) {
      continue;
    }
    if (IsValueNode<FuncGraph>(cnode->input(1))) {
      auto graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
      MS_LOG(DEBUG) << "Find the forward graph success";
      graph_set.insert(graph);
      auto manager = graph->manager();
      MS_EXCEPTION_IF_NULL(manager);
      auto graph_used = manager->func_graphs_used_total(graph);
      for (auto &sub_graph : graph_used) {
        graph_set.insert(sub_graph);
      }
    }
  }
  return graph_set;
}

void StepSplitSens(const std::pair<CNodePtr, LossNodeInfo> &sens_loss_pair) {
  CNodePtr sens_node = sens_loss_pair.first;
  auto loss_node = sens_loss_pair.second;
  auto loss_grad_layout = GetLossNodeGradOutputLayout(loss_node);
  if (!loss_grad_layout.empty()) {
    SplitSens(sens_node, loss_grad_layout[0]);
  }
}

bool IsPynativeParallel() {
  auto parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  auto execution_mode = MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE);
  return (execution_mode == kPynativeMode) && (parallel_mode == kSemiAutoParallel || parallel_mode == kAutoParallel);
}

// Sens node satisfies the following conditions: cnode(sens)-->cnode(tuple_getitem)-->cnode-->cnode(J)
std::vector<std::pair<CNodePtr, LossNodeInfo>> GetSensLossPairs(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  std::vector<std::pair<CNodePtr, LossNodeInfo>> sens_loss_pairs;
  for (auto &node : root->nodes()) {
    if (!node->isa<CNode>()) {
      continue;
    }

    // cnode(sens)-->cnode(tuple_getitem)
    auto sens_cnode = node->cast<CNodePtr>();
    AnfNodePtr expect_tuple_getitem = sens_cnode->input(0);
    MS_EXCEPTION_IF_NULL(expect_tuple_getitem);
    if (!expect_tuple_getitem->isa<CNode>()) {
      continue;
    }

    auto expect_tuple_getitem_cnode = expect_tuple_getitem->cast<CNodePtr>();
    if (!IsSomePrimitive(expect_tuple_getitem_cnode, prim::kTupleGetItem)) {
      continue;
    }

    // cnode(sens)-->cnode(tuple_getitem)-->cnode
    AnfNodePtr expect_anonymous = expect_tuple_getitem_cnode->input(1);
    MS_EXCEPTION_IF_NULL(expect_anonymous);
    if (!expect_anonymous->isa<CNode>()) {
      continue;
    }

    // cnode(sens)-->cnode(tuple_getitem)-->cnode-->cnode(J)
    auto expect_anonymous_cnode = expect_anonymous->cast<CNodePtr>();
    AnfNodePtr expect_j = expect_anonymous_cnode->input(0);
    MS_EXCEPTION_IF_NULL(expect_j);
    if (!expect_j->isa<CNode>()) {
      continue;
    }
    auto expect_j_cnode = expect_j->cast<CNodePtr>();
    if (!IsSomePrimitive(expect_j_cnode, J)) {
      continue;
    }

    if (!IsValueNode<FuncGraph>(expect_j_cnode->input(1))) {
      MS_LOG(EXCEPTION) << "Sens can't find the corresponding graph.";
    }
    auto func_graph = GetValueNode<FuncGraphPtr>(expect_j_cnode->input(1));
    auto loss_node_info = FindLossCNode(func_graph, 0);
    if (loss_node_info.loss_node == nullptr) {
      MS_LOG(WARNING) << "Can not find the loss cnode";
      continue;
    }
    std::pair<CNodePtr, LossNodeInfo> sens_loss_pair = std::make_pair(sens_cnode, loss_node_info);
    sens_loss_pairs.push_back(sens_loss_pair);
  }
  return sens_loss_pairs;
}

void ParallelCommunication(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes,
                           const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(manager);
  TensorRedistribution tensor_redistribution;

  std::vector<std::pair<CNodePtr, LossNodeInfo>> sens_loss_pairs = GetSensLossPairs(root);
  bool has_backward = !sens_loss_pairs.empty();
  // split sens must before inserting the operators.
  for (auto &pair : sens_loss_pairs) {
    // If the shape of grad-sens tensor is not [] or [1], use get tensor slice to handle it.
    // If the type of sens node is not Tensor, it is unsupported now, do nothing default.
    if (IsLastStage()) {
      StepSplitSens(pair);
    }
  }

  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      // the make_tuple is parallel care node, but it may have not operator info
      if (!IsParallelCareNode(cnode) || !cnode->has_user_data<OperatorInfo>()) {
        continue;
      }

      OperatorInfoPtr distribute_operator = GetDistributeOperator(cnode);
      MS_EXCEPTION_IF_NULL(distribute_operator);

      // skip Send Receive
      if (!cnode->HasPrimalAttr(PIPELINE_PARAM)) {
        // insert forward ops
        InsertForwardOps(distribute_operator, cnode);

        // insert redistribution ops
        StepRedistribution(cnode, distribute_operator, cnode, tensor_redistribution, cnode);
      }
      // insert backward ops
      if (has_backward || IsPynativeParallel()) {
        BackwardCommunication(root, distribute_operator, cnode, sens_loss_pairs);
      }

      distribute_operator->ReplaceNodeInputOrAttrs();
    } else if (IsValueNode<Tensor>(node) || IsValueNode<ValueList>(node) || IsValueNode<ValueTuple>(node)) {
      StepSplitTensor(node, manager);
    }
  }

  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      if (!IsParallelCareNode(cnode) || !cnode->has_user_data<OperatorInfo>() || IsSomePrimitive(cnode, RECEIVE) ||
          IsSomePrimitive(cnode, SEND)) {
        continue;
      }

      OperatorInfoPtr distribute_operator = GetDistributeOperator(cnode);
      MS_EXCEPTION_IF_NULL(distribute_operator);
      // StepReplace
      StepReplace(distribute_operator, cnode);
    }
  }
}

bool IsCohesiveNode(const CNodePtr &cnode) {
  return IsPrimitiveCNode(cnode, prim::kPrimCast) || IsPrimitiveCNode(cnode, prim::kPrimLoad) ||
         IsPrimitiveCNode(cnode, prim::kPrimAllGather) || IsPrimitiveCNode(cnode, prim::kPrimMiniStepAllGather) ||
         IsPrimitiveCNode(cnode, prim::kPrimMicroStepAllGather);
}

ParameterMap NodeParameterName(const CNodePtr &node, int64_t index, size_t curr_depth) {
  if (curr_depth > MAX_RECURSIVE_DEPTH) {
    MS_LOG(WARNING) << "When finding the parameters' name of a operator, exceeded the maximum depth: "
                    << MAX_RECURSIVE_DEPTH;
    return {};
  }
  std::vector<AnfNodePtr> node_inputs{node->inputs()};
  ParameterMap param_names;
  for (int64_t i = 0; i < UlongToLong(node_inputs.size()); ++i) {
    int64_t idx = index > i ? index : i;
    auto input = node_inputs[LongToSize(i)];
    if (input->isa<Parameter>()) {
      auto input_parameter = input->cast<ParameterPtr>();
      if (input_parameter->has_default() && ParameterRequireGrad(input_parameter)) {
        (void)param_names.emplace_back(std::make_pair(input_parameter->name(), input_parameter));
      }
    } else if (input->isa<CNode>()) {
      CNodePtr cnode = input->cast<CNodePtr>();
      if (!IsValueNode<Primitive>(cnode->input(0))) {
        continue;
      }
      if (IsCohesiveNode(cnode) && cnode->inputs().size() >= 1) {
        auto input_param_names = NodeParameterName(cnode, idx, 0);
        param_names.insert(param_names.end(), input_param_names.begin(), input_param_names.end());
      }
    }
  }
  return param_names;
}

bool IsGatherInfo(const std::string &name) {
  std::vector<std::string> gather_info_names = {"GatherInfo", "SparseGatherV2Info", "EmbeddingLookupInfo"};
  for (std::string info_name : gather_info_names) {
    if (name.find(info_name) != std::string::npos) {
      return true;
    }
  }
  return false;
}

void CheckpointStrategy(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root) {
  StrategyMap stra_map;
  TensorInfoMap tensor_info_map;
  ManualShapeMap manual_shape_map;
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    auto cnode = node->cast<CNodePtr>();
    if ((cnode == nullptr) || !IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    auto param_names = NodeParameterName(cnode, -1, 0);
    if (param_names.empty()) {
      continue;
    }
    string param_name = param_names[0].first;
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_EXCEPTION_IF_NULL(prim);
    OperatorInfoPtr operator_info = cnode->user_data<OperatorInfo>();
    if (operator_info) {
      if (operator_info->name().find(RESHAPEINFO) != std::string::npos) {
        continue;
      }
      std::string stratey_key_name = prim->name() + "_" + param_name;
      stra_map[stratey_key_name] = operator_info->strategy();
      for (auto param_name_pair : param_names) {
        tensor_info_map[param_name_pair.first] = param_name_pair.second->user_data<TensorLayout>();
      }
      if (IsGatherInfo(operator_info->name())) {
        auto gather_info = std::dynamic_pointer_cast<GatherInfo>(operator_info);
        auto param_split_shapes = gather_info->param_split_shapes();
        auto index_offsets = gather_info->index_offsets();
        if (param_split_shapes.size() != index_offsets.size()) {
          MS_LOG(EXCEPTION) << "In manual split, the param_split_shapes and index_offsets length should be same.";
        }
        std::vector<std::pair<int64_t, int64_t>> manual_shape;
        for (int64_t i = 0; i < UlongToLong(param_split_shapes.size()); ++i) {
          (void)manual_shape.emplace_back(
            std::make_pair(param_split_shapes[LongToSize(i)], index_offsets[LongToSize(i)]));
        }
        manual_shape_map[param_name] = manual_shape;
      }
    }
  }
  for (auto &cloned_parameter_node : root->parameters()) {
    MS_EXCEPTION_IF_NULL(cloned_parameter_node);
    auto cloned_parameter = cloned_parameter_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(cloned_parameter);

    if (!ParameterIsCloned(cloned_parameter_node)) {
      continue;
    }
    std::string cloned_param_name = cloned_parameter_node->cast<ParameterPtr>()->name();
    auto cloned_param_layout = cloned_parameter_node->user_data<TensorLayout>();
    if (cloned_param_layout == nullptr) {
      continue;
    }
    tensor_info_map[cloned_param_name] = cloned_param_layout;
  }
  if (StrategyCheckpoint::GetInstance().Save(stra_map, tensor_info_map, &manual_shape_map) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Save strategy checkpoint failed";
  }
}

void SetForwardFlag(const std::vector<AnfNodePtr> &all_nodes) {
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }

    // CNode is globally unique.
    MS_LOG(DEBUG) << "Set forward flag " << cnode->DebugString() << ".";
    cnode->set_in_forward_flag(true);
  }
}

void SetForwardFlag(const AnfNodeSet &all_nodes) {
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }

    // CNode is globally unique.
    cnode->set_in_forward_flag(true);
  }
}

std::set<FuncGraphPtr> ForwardGraph(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  const auto &all_nodes = root->nodes();
  std::set<FuncGraphPtr> graph_set = FindForwardGraphByRootNodes(all_nodes);
  return graph_set;
}

std::vector<AnfNodePtr> FindRootForwardCNode(const FuncGraphPtr &graph, const AnfNodeSet &all_nodes) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> root_forward_nodes;
  auto loss_cnode = FindLossCNode(graph, 0).loss_node;
  if (loss_cnode == nullptr) {
    MS_LOG(WARNING) << "Can not find the loss cnode";
    return root_forward_nodes;
  }

  auto loss_cnode_id = loss_cnode->UniqueIdThroughCopy();
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto root_node_id = node->UniqueIdThroughCopy();
    if (loss_cnode_id == root_node_id) {
      root_forward_nodes = DeepLinkedGraphSearch(cnode);
      break;
    }
  }
  return root_forward_nodes;
}

void InsertShapeOp(const CNodePtr &node, const AnfNodePtr &pre_node, const FuncGraphPtr &root) {
  // shape op doesn't have params and attrs.
  OperatorParams params;
  OperatorAttrs attrs;
  auto shape_value = GetValueNode(node->input(2))->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(shape_value);
  auto shape = shape_value->value();
  if (shape.empty()) {
    return;
  }
  OperatorArgs args = std::make_pair(attrs, params);
  Operator op = std::make_pair(SHAPE_OP, args);
  InsertNode(op, node, 2, pre_node, root, "shape");
}

static AnfNodePtr FindGrad(const CNodePtr &cnode, size_t curr_depth) {
  if (curr_depth > MAX_RECURSIVE_DEPTH) {
    MS_LOG(WARNING) << "When finding Grad nodes, exceeded the maximum recursion depth: " << MAX_RECURSIVE_DEPTH;
    return nullptr;
  }
  for (auto &node : cnode->inputs()) {
    if (!node->isa<CNode>()) {
      continue;
    }
    if (!IsPrimitiveCNode(node, prim::kPrimEnvironGet)) {
      return FindGrad(node->cast<CNodePtr>(), ++curr_depth);
    } else {
      return node;
    }
  }
  return nullptr;
}

void HandleRootReshapeAndSaveStrategy(const std::vector<AnfNodePtr> &all_nodes) {
  // If root graph has reshape op. Find the corresponding parameter.
  // Reshape's shape is the shape of the parameter.
  auto executor = pipeline::GraphExecutorPy::GetInstance();
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!IsValueNode<Primitive>(cnode->input(0)) || cnode == nullptr) {
      continue;
    }
    if (cnode->in_forward_flag()) {
      // Save strategy in executor
      OperatorInfoPtr op_info = cnode->user_data<OperatorInfo>();
      if (op_info) {
        auto stra_ptr = op_info->strategy();
        if (stra_ptr) {
          auto strategy = stra_ptr->GetInputDim();
          // fullname with scope should be found in step parallel end ir
          executor->SetCNodeStrategy(cnode->fullname_with_scope(), strategy);
        }
      }
      continue;
    }

    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim->name() != RESHAPE) {
      continue;
    }

    Shape origin_dst_shape = GetValue<std::vector<int64_t>>(cnode->input(2)->cast<ValueNodePtr>()->value());
    if (origin_dst_shape.size() == 1 && origin_dst_shape[0] == -1) {
      continue;
    }
    auto root = node->func_graph();
    auto grad_node = FindGrad(cnode, 0);
    if (grad_node) {
      InsertShapeOp(cnode, grad_node, root);
    }
  }
}

void MarkForwardCNode(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  auto all_nodes = root->nodes();
  auto graph_set = FindForwardGraphByRootNodes(all_nodes);

  if (graph_set.empty()) {
    MS_LOG(INFO) << "Can not find the forward graph, so mark the ops in root graph";
    SetForwardFlag(all_nodes);
  } else {
    for (auto &func_graph : graph_set) {
      MS_LOG(INFO) << "The sub graph size of root is " << root->func_graphs_used().size();
      auto return_node = func_graph->get_return();
      MS_EXCEPTION_IF_NULL(return_node);
      auto all_dfs_nodes = DeepLinkedGraphSearch(return_node);
      SetForwardFlag(all_dfs_nodes);
      auto root_forward_nodes = FindRootForwardCNode(func_graph, all_nodes);
      if (root_forward_nodes.empty()) {
        continue;
      }
      // Mark forward flag for the nodes in root graph.
      SetForwardFlag(root_forward_nodes);
    }
  }
}

CommInfo GetCommInfo() {
  int64_t device_num = ParallelContext::GetInstance()->device_num();
  int64_t global_rank = ParallelContext::GetInstance()->global_rank();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string backend = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  std::string world_group;
  std::string communication_backend;
  if (backend == kAscendDevice || backend == kDavinciDevice) {
    world_group = HCCL_WORLD_GROUP;
    communication_backend = HCCL_BACKEND;
  } else if (backend == kGPUDevice) {
    world_group = NCCL_WORLD_GROUP;
    communication_backend = NCCL_BACKEND;
  } else {
    MS_LOG(EXCEPTION) << "Invalid communication backend: " << backend;
  }
  uint32_t world_rank_size = 0;
  if (!CommManager::GetInstance().GetRankSize(world_group, &world_rank_size)) {
    MS_LOG(EXCEPTION) << "Get rank size failed";
  }

  if (!ParallelContext::GetInstance()->device_num_is_set()) {
    device_num = UintToInt(world_rank_size);
    MS_LOG(INFO) << "Get device num from communication model, the device num is  " << device_num;
  }
#if ENABLE_D || ENABLE_GPU
  if (ParallelContext::GetInstance()->device_num_is_set() && world_rank_size != device_num &&
      !ParallelContext::GetInstance()->hccl_test_available()) {
    // hccl_test_available is used when we compile graphs in real ascend card environment, but with hccl_test.
    MS_LOG(EXCEPTION) << "The device_num " << device_num << " set in the context is not consist with "
                      << world_rank_size << " devices you have"
                      << ". Please check your rank_table file(for Ascend) or host file(for GPU).";
  }
#endif
  uint32_t rank_id = 0;
  if (!ParallelContext::GetInstance()->global_rank_is_set()) {
    if (!CommManager::GetInstance().GetRankID(world_group, &rank_id)) {
      MS_LOG(EXCEPTION) << "Get rank id failed";
    }
    global_rank = UintToInt(rank_id);
    MS_LOG(INFO) << "Get global rank from communication model, the global rank is  " << global_rank;
  }
  CommInfo comm_info{device_num, global_rank, world_group, communication_backend};
  return comm_info;
}

Status ParallelInit() {
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  int32_t split_stage_num = ParallelContext::GetInstance()->pipeline_stage_split_num();
  std::string parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  if (split_stage_num <= 0) {
    MS_LOG(ERROR) << "The parameter 'split_stage_num' must be a positive number, but got the value : "
                  << split_stage_num;
    return FAILED;
  }
  auto comm_info = GetCommInfo();
  int64_t device_num = comm_info.device_num;
  int64_t global_rank = comm_info.global_rank;
  if ((device_num <= 0) || (device_num > MAX_DEVICE_NUM)) {
    MS_LOG(ERROR) << "The context configuration parameter 'device_num' must be positive, "
                     "but got the value of device_num: "
                  << device_num;
    return FAILED;
  }

  // the device_num maybe get from communication interface
  if (device_num % split_stage_num != 0) {
    MS_LOG(ERROR) << "The parameter 'device_num' must be divided by 'split_stage_num', but got the device_num : "
                  << device_num << "and the split_stage_num : " << split_stage_num;
    return FAILED;
  }

  if ((global_rank < 0) || (global_rank >= device_num)) {
    MS_LOG(ERROR) << "The parameter 'global_rank' must be  greater than 0 and less equal 'device num', "
                     "but got the global_rank : "
                  << global_rank << "and the device_num : " << device_num;
    return FAILED;
  }

  std::vector<int64_t> stages;
  for (int i = 0; i < split_stage_num; i++) {
    stages.push_back(device_num / split_stage_num);
  }

  bool use_sp = (ParallelContext::GetInstance()->strategy_search_mode() == kShardingPropagation) ||
                (ParallelContext::GetInstance()->sharding_propagation());
  if ((split_stage_num > 1) && (parallel_mode == kAutoParallel) && !use_sp) {
    MS_LOG(ERROR) << "To enable the pipeline parallel, please set the parallel mode to " << kSemiAutoParallel << " or "
                  << kAutoParallel << " with " << kShardingPropagation;
    return FAILED;
  }

  if (!InitDevice(device_num, global_rank, comm_info.communication_backend, stages)) {
    MS_LOG(ERROR) << "Init device failed";
    return FAILED;
  }

  MS_LOG(INFO) << "The parallel context: device_num: " << device_num << ", global_rank: " << global_rank
               << ", communication_backend: " << comm_info.communication_backend
               << ", gradients_mean: " << ParallelContext::GetInstance()->gradients_mean()
               << ", gradient_fp32_sync: " << ParallelContext::GetInstance()->gradient_fp32_sync();

  return SUCCESS;
}

void HandleForwardMakeTupleAndMakeList(const std::vector<AnfNodePtr> &all_nodes) {
  for (auto &node : all_nodes) {
    if (!AnfNodeIsPrimitive(node, MAKE_TUPLE) && !AnfNodeIsPrimitive(node, MAKE_LIST)) {
      continue;
    }

    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!cnode->in_forward_flag()) {
      continue;
    }

    FuncGraphManagerPtr manager = cnode->func_graph()->manager();
    MS_EXCEPTION_IF_NULL(manager);

    // MakeTuple has multiple users, each user's TensorInfo must be same.
    auto make_tuple_list_next_node = CheckMakeTupleSplit(node, manager);
    if (make_tuple_list_next_node == nullptr) {
      continue;
    }
    auto make_tuple_list_next_cnode = make_tuple_list_next_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple_list_next_cnode);
    OperatorInfoPtr op_info = GetDistributeOperator(make_tuple_list_next_cnode);
    MS_EXCEPTION_IF_NULL(op_info);
    cnode->set_user_data<OperatorInfo>(op_info);
  }
}

bool CreateGroupsByCkptFile(const std::string &file) {
  GroupInfoMap group_info_map;
  if (StrategyCheckpoint::GetInstance().LoadGroupInfo(file, &group_info_map) != SUCCESS) {
    return false;
  }

  if (CreateGroups(group_info_map) != SUCCESS) {
    return false;
  }
  MS_LOG(INFO) << "Create groups by checkpoint file success";
  return true;
}

void ReorderForPipelineSplit(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager, int64_t pipeline_stages) {
  if (!root->has_flag(BACKWARD) && pipeline_stages > 1) {
    root->set_flag(BACKWARD, true);
    if (root->has_flag(kTraining)) {
      Reorder(root);
    } else {
      ReorderForPredict(root, manager);
    }
  }
}

bool IsInsertVirtualOutput(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  auto comm_info = GetCommInfo();
  int64_t split_stage_num = ParallelContext::GetInstance()->pipeline_stage_split_num();
  int64_t per_stage_device_num = comm_info.device_num / split_stage_num;
  int64_t current_stage = comm_info.global_rank / per_stage_device_num;
  MS_LOG(INFO) << "The current stage is: " << current_stage;
  if (!root->has_flag(kTraining) && !ParallelContext::GetInstance()->dataset_strategy().empty()) {
    MS_LOG(WARNING) << "In eval/predict net, the output parallel strategy would not follow "
                       "the input parallel strategy when using context.set_auto_parallel_context(dataset_strategy)"
                       " to configure the input strategy.";
  }
  return ((!root->has_flag(kTraining) && ParallelContext::GetInstance()->dataset_strategy().empty() &&
           current_stage == split_stage_num - 1) ||
          IsPynativeParallel());
}

static void HandleGroupInfo(const FuncGraphPtr &root) {
  auto group_info = g_device_manager->group_info();
  auto group_info_save_path = common::GetEnv("GROUP_INFO_FILE");
  if (!group_info_save_path.empty()) {
    ParallelContext::GetInstance()->set_group_ckpt_save_file(group_info_save_path);
  }

  if (StrategyCheckpoint::GetInstance().group_info_save_on()) {
    RankList comm_group = FindCommonMirrorGroup(root);
    if (StrategyCheckpoint::GetInstance().SaveGroupInfo(group_info, comm_group) != SUCCESS) {
      MS_LOG(EXCEPTION) << "Save group info failed";
    }
  }
}

static void HandleDataParallel() {
  std::string parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode == kDataParallel) {
    auto group_info_save_path = common::GetEnv("GROUP_INFO_FILE");
    if (!group_info_save_path.empty()) {
      std::vector<std::pair<std::string, std::vector<uint32_t>>> group_info;
      int64_t device_num = GetCommInfo().device_num;
      RankList comm_group;
      for (size_t i = 0; i < size_t(device_num); ++i) {
        comm_group.push_back(i);
      }
      ParallelContext::GetInstance()->set_group_ckpt_save_file(group_info_save_path);
      if (StrategyCheckpoint::GetInstance().SaveGroupInfo(group_info, comm_group) != SUCCESS) {
        MS_LOG(EXCEPTION) << "Save group info failed";
      }
    }
  }
}

static void PipelinePreProcess(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager,
                               const std::vector<AnfNodePtr> &all_nodes) {
  auto pipeline_stages = ParallelContext::GetInstance()->pipeline_stage_split_num();
  if (pipeline_stages > 1) {
    HandleMicroBatch(all_nodes, manager);
    ParameterStartNode(all_nodes, manager);
    LastStageEndNode(all_nodes, manager, root);
  }
}

static void PipelinePostProcess(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes) {
  auto pipeline_stages = ParallelContext::GetInstance()->pipeline_stage_split_num();
  if (pipeline_stages > 1) {
    AddVirtualAssignAdd(root);
    HandleReceiveParam(root, all_nodes);
    LabelGenMaskMicro(root);
  }
}

static void InsertAllReduceForNormValue(const AnfNodePtr &res_node) {
  auto cnode = res_node->cast<CNodePtr>();
  auto graphs = res_node->func_graph();
  MS_EXCEPTION_IF_NULL(graphs);
  auto manager = graphs->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto node_user_map = manager->node_users();
  if (!IsSomePrimitive(cnode, EXPAND_DIMS)) {
    MS_LOG(ERROR) << "Expected the operator expand_dims, but found the " << GetPrimName(cnode)
                  << "This may cause the calculation of the global norm incorrect";
    return;
  }
  auto pipeline_stages = ParallelContext::GetInstance()->pipeline_stage_split_num();
  auto find_node = res_node;
  uint32_t limits = 0;
  while (!IsSomePrimitive(find_node->cast<CNodePtr>(), SQRT) && limits < MAX_BFS_DEPTH) {
    auto users = node_user_map.at(find_node);
    if (users.empty()) return;
    find_node = users.front().first;
    ++limits;
  }
  if (!find_node || !IsSomePrimitive(find_node->cast<CNodePtr>(), SQRT)) {
    return;
  }
  auto anf_node = find_node->cast<CNodePtr>();
  if (anf_node->inputs().size() > 1 && IsSomePrimitive(anf_node->input(1)->cast<CNodePtr>(), ALL_REDUCE)) {
    return;
  }
  auto sqrt_node = find_node;
  auto cur_stage_rank_list = g_device_manager->GetDeviceListInThisStage();
  Group cur_stage_device_list;
  if (g_device_manager->CreateGroup(cur_stage_rank_list, &cur_stage_device_list) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Create the communication group for allreduce in calculating global norm failed, "
                         "the rank_list is: "
                      << cur_stage_rank_list;
  }
  InsertAllReduceToNodeInput(sqrt_node->cast<CNodePtr>(), cur_stage_device_list.name(), PARALLEL_GLOBALNORM);
  MS_LOG(INFO) << "Insert the AllReduce for global norm value in stages succeed.";
  if (pipeline_stages > 1) {
    MS_LOG(INFO) << "Insert the AllReduce for global norm value between stages succeed.";
    auto ranks_between_stages = g_device_manager->GetDeviceListBetweenStage();
    Group group_between_stages;
    if (g_device_manager->CreateGroup(ranks_between_stages, &group_between_stages)) {
      MS_LOG(EXCEPTION) << "Create the communication group for allreduce in calculating global norm "
                           "with pipeline parallel failed, the rank_list is: "
                        << cur_stage_rank_list;
    }
    InsertAllReduceToNodeInput(sqrt_node->cast<CNodePtr>(), group_between_stages.name(), PARALLEL_GLOBALNORM_BETWEEN);
  }
}

AnfNodePtr FindExpanDimsWIthGradScale(const AnfNodePtr &node_ptr, const NodeUsersMap &node_users_map, uint32_t limits) {
  std::queue<AnfNodePtr> visited;
  AnfNodePtr queue_node = nullptr;
  CNodePtr cnode = nullptr;
  AnfNodePtr last_node = nullptr;
  uint32_t depth = 0;
  if (!node_ptr) {
    return nullptr;
  }
  visited.push(node_ptr);
  while (!visited.empty()) {
    queue_node = visited.front();
    visited.pop();
    cnode = queue_node->cast<CNodePtr>();
    // MAKE_TUPLE will not appear after the load in the forward graph
    if (IsSomePrimitive(cnode, EXPAND_DIMS)) {
      auto value = GetAttrsFromAnfNode(queue_node, GRAD_SCALE);
      if (!value || !GetValue<bool>(value)) {
        continue;
      }
      return queue_node;
    }
    if (!IsSomePrimitiveList(cnode, {ENVIRONGET, MUL, SQUARE, REDUCE_SUM, EXPAND_DIMS, DEPEND, CAST, REF_TO_EMBED})) {
      continue;
    }
    auto node_set = node_users_map.at(queue_node);
    for (auto &node_user : node_set) {
      visited.push(node_user.first);
    }
    if (!last_node || last_node == queue_node) {
      if (++depth == limits) {
        break;
      }
      last_node = visited.back();
    }
  }
  return nullptr;
}

static void InsertDivAndAllReduceForNorm(const NodeUsersMap &node_user_map, const AnfNodePtr &parameter,
                                         uint32_t dev_num) {
  auto params_user_set = node_user_map.at(parameter);
  for (auto &param_pair : params_user_set) {
    auto cnode = param_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->in_forward_flag()) {
      continue;
    }
    auto expand_dims_node = FindExpanDimsWIthGradScale(cnode, node_user_map, MAX_BFS_DEPTH);
    if (!expand_dims_node) continue;
    auto value = GetAttrsFromAnfNode(expand_dims_node, GRAD_SCALE);
    if (!value || !GetValue<bool>(value)) continue;
    if (dev_num > 0) {
      InsertRealDivOpToNodeInput(expand_dims_node->cast<CNodePtr>(), dev_num, PARALLEL_GLOBALNORM_DIV);
      MS_LOG(INFO) << "Insert the realdiv with " << dev_num << " for the parameter " << parameter->fullname_with_scope()
                   << "succeed!";
    }
    // If already inserted allreduce, the pattern will not be matched and thus no allreduce will be inserted.
    InsertAllReduceForNormValue(expand_dims_node);
  }
}

static AnfNodePtr GetMirrorOp(const NodeUsersMap &node_user_map, const AnfNodePtr &parameter) {
  auto params_user_set = node_user_map.at(parameter);
  for (auto &param_pair : params_user_set) {
    auto cnode = param_pair.first->cast<CNodePtr>();
    std::vector<AnfNodePtr> candidate = {cnode};
    if (!cnode->in_forward_flag()) {
      continue;
    }
    if (IsInTrivialNodeList(cnode) || IsSomePrimitive(cnode, LOAD)) {
      auto load_users = node_user_map.at(param_pair.first);
      std::transform(load_users.begin(), load_users.end(), std::back_inserter(candidate),
                     [](const auto &v) { return v.first; });
    }
    for (auto &node : candidate) {
      auto local_cnode = node->cast<CNodePtr>();
      if (!IsPrimitiveCNode(local_cnode, prim::kPrimMirror) &&
          !IsPrimitiveCNode(local_cnode, prim::kPrimMirrorMicroStep) &&
          !IsPrimitiveCNode(local_cnode, prim::kPrimMirrorMiniStep)) {
        continue;
      }
      return node;
    }
  }
  return nullptr;
}

static void HandlGlobalNormScale(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes,
                                 const FuncGraphManagerPtr &manager) {
  auto parameters = root->parameters();
  auto node_user_map = manager->node_users();
  MS_LOG(INFO) << "Start to process the global norm";

  for (auto &parameter : parameters) {
    int64_t dev_num = 0;
    if (!ParameterRequireGrad(parameter)) continue;
    auto mirror_node = GetMirrorOp(node_user_map, parameter);
    auto device_num_ptr = GetAttrsFromAnfNode(mirror_node, DEV_NUM);
    if (device_num_ptr && device_num_ptr->isa<Int64Imm>()) {
      dev_num = GetValue<int64_t>(device_num_ptr);
    }
    InsertDivAndAllReduceForNorm(node_user_map, parameter, dev_num);
  }
}

bool StepParallel(const FuncGraphPtr &root, const opt::OptimizerPtr &optimizer) {
#if ((defined ENABLE_CPU) && (!defined _WIN32) && !defined(__APPLE__))
  if (ps::PSContext::instance()->is_server() || ps::PSContext::instance()->is_scheduler()) {
    return false;
  }
#endif
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(optimizer);
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  std::string parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  HandleDataParallel();
  pipeline::ResourceBasePtr res = optimizer->resource();
  MS_EXCEPTION_IF_NULL(res);
  FuncGraphManagerPtr manager = res->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto pipeline_stages = ParallelContext::GetInstance()->pipeline_stage_split_num();
  // assume no change to graph
  bool changes = false;
  // control whether use model_parallel mode
  if (!root->has_flag(kAutoParallel) || ((parallel_mode != kAutoParallel) && (parallel_mode != kSemiAutoParallel)) ||
      (root->has_flag(SEMI_AUTO_PARALLEL_RUN_ONCE_ONLY))) {
    if (!root->has_flag(CHECK_SET_STRATEGY_VALID_ONCE_ONLY)) {
      MS_LOG(INFO) << "Strategies would be ignored in " << parallel_mode
                   << ", shard() only valid in [semi_]auto_parallel.";
      root->set_flag(CHECK_SET_STRATEGY_VALID_ONCE_ONLY, true);
    }
    ReorderForPipelineSplit(root, manager, pipeline_stages);

    return changes;
  }

  struct timeval start_time, end_time;
  (void)gettimeofday(&start_time, nullptr);

  MS_LOG(INFO) << "Now entering step parallel";
  DumpGraph(root, std::string(STEP_PARALLEL_BEGIN));
  AnfNodePtr ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  std::reverse(all_nodes.begin(), all_nodes.end());
  if (pipeline_stages <= 1 && parallel_mode != kAutoParallel && ParallelInit() != SUCCESS) {
    MS_LOG(EXCEPTION) << "Parallel init failed";
  }
  PipelinePreProcess(root, manager, all_nodes);
  if (parallel_mode != kAutoParallel) {
    TOTAL_OPS = 0;
    // mark the forward cnodes, parallel only care these nodes
    MarkForwardCNode(root);

    if (FindCommunicationOp(all_nodes)) {
      MS_LOG(EXCEPTION) << "The graph contain communication op";
    }

    if (IsInsertVirtualOutput(root)) {
      InsertVirtualOutput(root, all_nodes);
      AnfNodePtr ret_after = root->get_return();
      MS_EXCEPTION_IF_NULL(ret_after);
      all_nodes = DeepScopedGraphSearch(ret_after);
      std::reverse(all_nodes.begin(), all_nodes.end());
    }

    // extract shape and strategy, set operator_info
    ExtractInformation(all_nodes);
    ReshapeInit(all_nodes);
  }

  SetCastForParamNotRecompute(all_nodes);

  HandleRootReshapeAndSaveStrategy(all_nodes);

  HandleForwardMakeTupleAndMakeList(all_nodes);

  // if the input or parameter has multiple users, check whether its split strategies are consistent.
  CheckParameterSplit(all_nodes);

  HandleSymbolicKeyInstance(root, all_nodes);

  // cover Parallel shape
  CoverSliceShape(root);

  // handle input is not used
  HandleNoUsedParameter(root);

  // set the shape for optimizer's clone tensor
  SetClonedTensorShapeForOptimizer(root);

  HandleAdaFactorOpt(root);

  auto adasum_param_tensor_layout_map = AdaSumParamTensorLayout(root);
  bool is_apply_adasum = HandleAdaSum(root, all_nodes, &adasum_param_tensor_layout_map);

  // save strategy as checkpoint for multi-train
  if (StrategyCheckpoint::GetInstance().SaveCheckPointOn()) {
    CheckpointStrategy(all_nodes, root);
  }
  // ForwardCommunication BackwardCommunication TensorRedistribution
  ParallelCommunication(root, all_nodes, manager);
  if (is_apply_adasum) {
    HandleMirrorInAdaSum(root, &adasum_param_tensor_layout_map);
  }

  PipelinePostProcess(root, all_nodes);

  HandleGroupInfo(root);

  // handle full split parammeters in grad accumulation, do not contain optimizer-sharding's parameter
  HandleFullySplitParameters(root);

  HandlGlobalNormScale(root, all_nodes, manager);

  DumpGraph(root, std::string(STEP_PARALLEL_END));

  // step parallel only run once
  root->set_flag(SEMI_AUTO_PARALLEL_RUN_ONCE_ONLY, true);
  res->SetResult(pipeline::kStepParallelGraph, root);

  // in auto parallel mode, no need to check if stategies set
  root->set_flag(CHECK_SET_STRATEGY_VALID_ONCE_ONLY, true);

  (void)gettimeofday(&end_time, nullptr);
  uint64_t time = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  time += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);

  MS_LOG(INFO) << "Now leaving step parallel, used time: " << time << " us";
  return changes;
}

// Needed by rec_parser
std::vector<std::string> ExtractInputsTensorName(const CNodePtr &node) {
  std::vector<std::string> name_inputs;
  std::vector<AnfNodePtr> all_inputs = node->inputs();
  std::vector<AnfNodePtr> node_inputs{all_inputs.begin() + 1, all_inputs.end()};

  std::string node_id = node->UniqueId();
  name_inputs.push_back(node_id);
  for (auto &input : node_inputs) {
    std::string name = input->UniqueId();
    name_inputs.push_back(name);
  }

  return name_inputs;
}
}  // namespace parallel
}  // namespace mindspore
