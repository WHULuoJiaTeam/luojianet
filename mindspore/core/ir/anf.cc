/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#include "ir/anf.h"

#include <algorithm>
#include <sstream>
#include <vector>
#include <queue>

#include "utils/hash_map.h"
#include "base/core_ops.h"
#include "ir/func_graph.h"
#include "ir/primitive.h"
#include "utils/ms_context.h"
#include "utils/anf_utils.h"

namespace mindspore {
const AbstractBasePtr &AnfNode::abstract() const {
  // cppcheck-suppress unreadVariable
  auto lock = AnfUtils::GetAbstractLock(this);
  return abstract_;
}

void AnfNode::set_abstract(const AbstractBasePtr &abs) {
  // cppcheck-suppress unreadVariable
  auto lock = AnfUtils::GetAbstractLock(this);
  abstract_ = abs;
}

// namespace to support intermediate representation definition
CNode::CNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &func_graph)
    : AnfNode(func_graph),
      inputs_(inputs),
      primal_attrs_(PrimalAttrManager::GetInstance().GetCurrentPrimalAttr()),
      primal_debug_infos_(PrimalDebugInfoManager::GetInstance().GetCurrentPrimalDebugInfo()) {}

CNode::CNode(std::vector<AnfNodePtr> &&inputs, const FuncGraphPtr &func_graph)
    : AnfNode(func_graph),
      inputs_(std::move(inputs)),
      primal_attrs_(PrimalAttrManager::GetInstance().GetCurrentPrimalAttr()),
      primal_debug_infos_(PrimalDebugInfoManager::GetInstance().GetCurrentPrimalDebugInfo()) {}

CNode::CNode(std::vector<AnfNodePtr> &&inputs, const FuncGraphPtr &func_graph, NodeDebugInfoPtr &&debug_info)
    : AnfNode(func_graph, std::move(debug_info)),
      inputs_(std::move(inputs)),
      primal_attrs_(PrimalAttrManager::GetInstance().GetCurrentPrimalAttr()),
      primal_debug_infos_(PrimalDebugInfoManager::GetInstance().GetCurrentPrimalDebugInfo()) {}

// Check if CNode is an apply with the specific Primitive.
bool CNode::IsApply(const PrimitivePtr &value) const {
  if (value == nullptr) {
    return false;
  }

  if (inputs_.size() != 0 && IsValueNode<Primitive>(inputs_[0])) {
    PrimitivePtr fn_value = GetValueNode<PrimitivePtr>(inputs_[0]);
    if (fn_value->Hash() == value->Hash() && fn_value->name() == value->name()) {
      return true;
    }
  }

  return false;
}

void CNode::add_input(const AnfNodePtr &input) {
  inputs_.push_back(input);
  input_tensor_num_ = -1;
}

void CNode::set_input(size_t i, const AnfNodePtr &new_input) {
  if (i >= inputs_.size()) {
    MS_LOG(EXCEPTION) << "i: " << i << " out of range: " << inputs_.size() << ", cnode: " << DebugString();
  }
  inputs_[i] = new_input;
  input_tensor_num_ = -1;
}

void CNode::set_inputs(const std::vector<AnfNodePtr> &inputs) {
  inputs_ = inputs;
  input_tensor_num_ = -1;
}

const AnfNodePtr &CNode::input(size_t i) const {
  if (i >= inputs_.size()) {
    MS_LOG(EXCEPTION) << "i: " << i << " out of range: " << inputs_.size() << ", cnode: " << DebugString();
  }
  return inputs_.at(i);
}

std::string CNode::DebugString(int recursive_level) const {
  std::ostringstream buffer;
  if (recursive_level > 0) {
    if (func_graph() != nullptr) {
      buffer << func_graph()->ToString() << ":";
    }
    buffer << ToString() << "{";
    bool is_first_node = true;
    int idx = 0;
    for (auto &node : inputs_) {
      MS_EXCEPTION_IF_NULL(node);
      if (is_first_node) {
        is_first_node = false;
      } else {
        buffer << ", ";
      }
      buffer << "[" << idx << "]: " << node->DebugString(recursive_level - 1);
      idx++;
    }
    buffer << "}";
  } else {
    buffer << ToString();
  }
  return buffer.str();
}

void CNode::AddFusedDebugInfo(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return;
  }
  if (shared_from_this() == node) {
    this->AddFusedDebugInfo(node->debug_info());
    return;
  }
  auto cnode = node->cast<CNodePtr>();
  auto node_fused_debug_infos = cnode->fused_debug_infos();
  if (!node_fused_debug_infos.empty()) {
    (void)std::for_each(node_fused_debug_infos.begin(), node_fused_debug_infos.end(),
                        [this](const NodeDebugInfoPtr &debug_info) { this->AddFusedDebugInfo(debug_info); });
  } else {
    this->AddFusedDebugInfo(cnode->debug_info());
  }

  auto primal_debug_infos = cnode->primal_debug_infos();
  if (!primal_debug_infos.empty()) {
    (void)std::for_each(primal_debug_infos.begin(), primal_debug_infos.end(),
                        [this](const NodeDebugInfoPtr &debug_info) { this->AddPrimalDebugInfo(debug_info); });
  }
}

void CNode::AddFusedDebugInfoList(const std::vector<AnfNodePtr> &nodes) {
  (void)std::for_each(nodes.begin(), nodes.end(), [this](const AnfNodePtr &node) { this->AddFusedDebugInfo(node); });
}

void CNode::AddFusedDebugInfo(const NodeDebugInfoPtr &debug_info) {
  if (debug_info == nullptr) {
    return;
  }
  (void)fused_debug_infos_.emplace(debug_info);
}

void CNode::AddFusedDebugInfoList(const std::vector<NodeDebugInfoPtr> &debug_infos) {
  (void)std::for_each(debug_infos.begin(), debug_infos.end(),
                      [this](const NodeDebugInfoPtr &debug_info) { this->AddFusedDebugInfo(debug_info); });
}

NodeDebugInfoSet CNode::primal_debug_infos() const { return primal_debug_infos_; }

void CNode::set_primal_debug_infos(const NodeDebugInfoSet &debug_infos) {
  (void)std::for_each(debug_infos.begin(), debug_infos.end(),
                      [this](const NodeDebugInfoPtr &debug_info) { this->AddPrimalDebugInfo(debug_info); });
}

void CNode::AddPrimalDebugInfo(const NodeDebugInfoPtr &debug_info) { (void)primal_debug_infos_.emplace(debug_info); }

std::string Parameter::DebugString(int recursive_level) const {
  std::ostringstream buffer;
  if (recursive_level > 0) {
    if (func_graph() != nullptr) {
      buffer << func_graph()->ToString() << ":";
    }
  }
  buffer << ToString();
  return buffer.str();
}

ParamInfoPtr Parameter::param_info() const {
  if (!has_default()) {
    return nullptr;
  }
  auto tensor = default_param()->cast<tensor::MetaTensorPtr>();
  if (tensor == nullptr || !tensor->is_parameter()) {
    return nullptr;
  }
  return tensor->param_info();
}

std::string ValueNode::ToString() const {
  MS_EXCEPTION_IF_NULL(value_);
  if (value_->isa<FuncGraph>()) {
    return value_->ToString();
  }
  std::ostringstream buffer;
  buffer << AnfNode::ToString();
  buffer << "(" << value_->ToString() << ")";
  return buffer.str();
}

std::string ValueNode::DebugString(int) const {
  MS_EXCEPTION_IF_NULL(value_);
  std::ostringstream buffer;
  buffer << "ValueNode<" << value_->type_name() << "> " << value_->ToString();
  return buffer.str();
}

std::string ValueNode::fullname_with_scope() {
  if (!fullname_with_scope_.empty()) {
    return fullname_with_scope_;
  }

  MS_EXCEPTION_IF_NULL(scope());
  fullname_with_scope_ = scope()->name() + "/" + "data-" + id_generator::get_id(shared_from_base<ValueNode>());
  return fullname_with_scope_;
}

bool IsPrimitiveCNode(const AnfNodePtr &node, const PrimitivePtr &value) {
  auto cnode = dyn_cast<CNode>(node);
  if (cnode == nullptr) {
    return false;
  }
  if (value != nullptr) {
    return cnode->IsApply(value);
  }
  const auto &prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  return prim != nullptr;
}

PrimitivePtr GetCNodePrimitive(const AnfNodePtr &node) {
  auto cnode = dyn_cast<CNode>(node);
  if (cnode != nullptr) {
    if (cnode->size() > 0) {
      auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
      return prim;
    }
  }
  return nullptr;
}

std::string GetCNodeFuncName(const CNodePtr cnode) {
  if (cnode->inputs().empty()) {
    return "";
  }

  AnfNodePtr valuenode = cnode->input(0);
  auto value = GetValueNode(valuenode);
  if (value != nullptr) {
    // check whether the valuenode is primitive
    if (value->isa<Primitive>()) {
      return value->cast<PrimitivePtr>()->name();
    }
    return value->ToString();
  }
  return "";
}

FuncGraphPtr GetCNodeFuncGraph(const AnfNodePtr &node) {
  auto cnode = dyn_cast<CNode>(node);
  if (cnode != nullptr && cnode->size() > 0) {
    return GetValueNode<FuncGraphPtr>(cnode->input(0));
  }
  return nullptr;
}

bool IsPrimitive(const AnfNodePtr &node, const PrimitivePtr &value) {
  if (IsValueNode<Primitive>(node)) {
    PrimitivePtr fn_value = GetValueNode<PrimitivePtr>(node);
    MS_EXCEPTION_IF_NULL(value);
    if (fn_value->Hash() == value->Hash() && fn_value->name() == value->name()) {
      return true;
    }
  }
  return false;
}

bool IsPrimitiveEquals(const PrimitivePtr &prim1, const PrimitivePtr &prim2) {
  if (prim1 == nullptr || prim2 == nullptr) {
    return false;
  }
  return (prim1 == prim2) || (prim1->Hash() == prim2->Hash() && prim1->name() == prim2->name());
}

size_t GetAbstractMonadNum(const AbstractBasePtrList &args) {
  size_t num = 0;
  for (auto &arg : args) {
    if (arg->isa<abstract::AbstractMonad>()) {
      ++num;
    }
  }
  return num;
}

template <typename T>
bool HasAbstract(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  const auto &abs = node->abstract();
  return (abs != nullptr && abs->isa<T>());
}

bool HasAbstractMonad(const AnfNodePtr &node) { return HasAbstract<abstract::AbstractMonad>(node); }

bool HasAbstractUMonad(const AnfNodePtr &node) { return HasAbstract<abstract::AbstractUMonad>(node); }

bool HasAbstractIOMonad(const AnfNodePtr &node) { return HasAbstract<abstract::AbstractIOMonad>(node); }

bool GetPrimitiveFlag(const PrimitivePtr &prim, const std::string &attr) {
  if (prim != nullptr) {
    auto flag = prim->GetAttr(attr);
    if (flag && flag->isa<BoolImm>()) {
      return GetValue<bool>(flag);
    }
  }
  return false;
}

EffectInfo GetPrimEffectInfo(const PrimitivePtr &prim) {
  bool mem = GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_MEM);
  bool io = GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_IO);
  return {EffectInfo::kDetected, mem, io, false};
}

std::set<CNodePtr> GetLoadInputs(const AnfNodePtr &node) {
  std::set<CNodePtr> loads;
  auto cnode = dyn_cast<CNode>(node);
  if (cnode == nullptr) {
    return loads;
  }
  auto &inputs = cnode->inputs();
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto &input = inputs.at(i);
    if (IsPrimitiveCNode(input, prim::kPrimLoad)) {
      loads.insert(input->cast<CNodePtr>());
    } else if (IsPrimitiveCNode(input, prim::kPrimMakeTuple)) {
      loads.merge(GetLoadInputs(input));
    }
  }
  return loads;
}

bool IsStateEquivalent(const AnfNodePtr &outer, const AnfNodePtr &inner) {
  constexpr size_t kMonadInput = 2;
  auto outer_loads = GetLoadInputs(outer);
  if (outer_loads.empty()) {
    return true;
  }
  auto inner_loads = GetLoadInputs(inner);
  if (inner_loads.empty()) {
    return true;
  }
  outer_loads.merge(inner_loads);
  auto &monad = (*outer_loads.begin())->inputs().at(kMonadInput);
  return std::all_of(++outer_loads.begin(), outer_loads.end(),
                     [&monad, kMonadInput](const CNodePtr &load) { return load->inputs().at(kMonadInput) == monad; });
}

SeenNum NewSeenGeneration() {
  static SeenNum seen_generation = 0;
  return ++seen_generation;
}

namespace id_generator {
static mindspore::HashMap<std::string, int> node_ids;
std::string get_id(const AnfNodePtr &node) {
  auto type_name = node->type_name();
  if (node_ids.find(type_name) == node_ids.end()) {
    node_ids[type_name] = 0;
  } else {
    node_ids[type_name]++;
  }
  return std::to_string(node_ids[type_name]);
}

void reset_id() { node_ids.clear(); }
}  // namespace id_generator
auto constexpr kTargetUnDefined = "kTargetUnDefined";
auto constexpr kPrimitiveTarget = "primitive_target";
namespace {
PrimitivePtr GetPrimitiveFromValueNode(const AnfNodePtr &node) {
  if (node == nullptr) {
    return nullptr;
  }
  auto value_node = node->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    return nullptr;
  }
  auto value = value_node->value();
  if (value == nullptr || !value->isa<Primitive>()) {
    return nullptr;
  }
  return value->cast<PrimitivePtr>();
}

static std::string GetNodeTargetForVarInputNode(const CNodePtr &cnode) {
  auto &inputs = cnode->inputs();
  std::vector<AnfNodePtr> real_inputs;
  const size_t update_state_valid_input_index = 2;
  const size_t make_tuple_valid_input_index = 1;
  if (cnode->IsApply(prim::kPrimUpdateState) && inputs.size() > update_state_valid_input_index) {
    (void)std::copy(inputs.begin() + SizeToLong(update_state_valid_input_index), inputs.end(),
                    std::back_inserter(real_inputs));
  } else if (cnode->IsApply(prim::kPrimMakeTuple) && inputs.size() > make_tuple_valid_input_index) {
    (void)std::copy(inputs.begin() + SizeToLong(make_tuple_valid_input_index), inputs.end(),
                    std::back_inserter(real_inputs));
  }
  std::string first_input_target = kTargetUnDefined;
  bool has_diff_target =
    std::any_of(std::rbegin(real_inputs), std::rend(real_inputs), [&first_input_target](const AnfNodePtr &n) {
      auto target = GetOriginNodeTarget(n);
      if (target == kTargetUnDefined) {
        return false;
      }
      if (first_input_target == kTargetUnDefined) {
        first_input_target = target;
      }
      return target != first_input_target;
    });
  if (!has_diff_target) {
    return first_input_target;
  }
  return kTargetUnDefined;
}

static inline bool IsSummaryPrimitiveCNode(const AnfNodePtr &node) {
  return IsPrimitiveCNode(node, prim::kPrimImageSummary) || IsPrimitiveCNode(node, prim::kPrimScalarSummary) ||
         IsPrimitiveCNode(node, prim::kPrimTensorSummary) || IsPrimitiveCNode(node, prim::kPrimHistogramSummary);
}

std::string GetVirtualNodeTargetFromInputs(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto &inputs = cnode->inputs();
#ifndef ENABLE_SECURITY
  if (IsSummaryPrimitiveCNode(node)) {
    if (inputs.size() > 1) {
      return GetOriginNodeTarget(inputs[1]);
    }
    return kTargetUnDefined;
  }
#endif
  if (IsPrimitiveCNode(node, prim::kPrimDepend) || IsPrimitiveCNode(node, prim::kPrimLoad)) {
    const size_t node_inputs_num = 3;
    if (inputs.size() >= node_inputs_num) {
      size_t use_index = 1;
      if (!inputs[use_index]->isa<CNode>()) {
        use_index = 2;
      }
      return GetOriginNodeTarget(inputs[use_index]);
    }
  } else if (IsPrimitiveCNode(node, prim::kPrimMakeTuple) || IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
    return GetNodeTargetForVarInputNode(node->cast<CNodePtr>());
  } else if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    return GetOriginNodeTarget(cnode->input(1));
  }
  return kTargetUnDefined;
}

std::string GetVirtualNodeTargetFromUsers(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto func_graph = cnode->func_graph();
  if (func_graph == nullptr) {
    return kTargetUnDefined;
  }
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    return kTargetUnDefined;
  }
  auto users = manager->node_users()[cnode];
  std::string first_user_target = kTargetUnDefined;
  bool has_diff_target =
    std::any_of(std::begin(users), std::end(users), [&first_user_target](const std::pair<AnfNodePtr, int> &u) {
      auto target = GetOriginNodeTarget(u.first);
      if (target == kTargetUnDefined) {
        return false;
      }
      if (first_user_target == kTargetUnDefined) {
        first_user_target = target;
      }
      return target != first_user_target;
    });
  if (!has_diff_target) {
    return first_user_target;
  }
  return kTargetUnDefined;
}

std::string GetVirtualNodeTarget(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  node->set_user_data(kPrimitiveTarget, std::make_shared<std::string>(kTargetUnDefined));
  auto target = GetVirtualNodeTargetFromInputs(node);
  node->set_user_data(kPrimitiveTarget, std::make_shared<std::string>(target));
  if (target != kTargetUnDefined) {
    return target;
  }
  target = GetVirtualNodeTargetFromUsers(node);
  node->set_user_data(kPrimitiveTarget, std::make_shared<std::string>(target));
  return target;
}

std::string GetTargetFromAttr(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto attr_input = cnode->input(0);
  auto primitive = GetPrimitiveFromValueNode(attr_input);
  if (primitive == nullptr) {
    return kTargetUnDefined;
  }
  auto att_target = primitive->GetAttr(kPrimitiveTarget);
  if (att_target != nullptr) {
    if (!att_target->isa<StringImm>()) {
      MS_LOG(EXCEPTION) << "Only support string CPU|GPU|Ascend for primitive_target";
    }
    auto target = GetValue<std::string>(att_target);
    if (kTargetSet.find(target) == kTargetSet.end()) {
      MS_LOG(EXCEPTION) << "Only support string CPU|GPU|Ascend for primitive_target, but get " << target;
    }
    return target;
  }
  return kTargetUnDefined;
}
}  // namespace

std::string GetOriginNodeTarget(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return kTargetUnDefined;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto ud_target = cnode->user_data<std::string>(kPrimitiveTarget);
  if (ud_target != nullptr) {
    return *ud_target.get();
  }
  auto target = GetTargetFromAttr(node);
  if (target != kTargetUnDefined) {
    return target;
  }
#ifndef ENABLE_SECURITY
  if (IsPrimitiveCNode(node, prim::kPrimImageSummary) || IsPrimitiveCNode(node, prim::kPrimScalarSummary) ||
      IsPrimitiveCNode(node, prim::kPrimTensorSummary) || IsPrimitiveCNode(node, prim::kPrimHistogramSummary) ||
      IsPrimitiveCNode(node, prim::kPrimDepend) || IsPrimitiveCNode(node, prim::kPrimLoad) ||
      IsPrimitiveCNode(node, prim::kPrimUpdateState) || IsPrimitiveCNode(node, prim::kPrimMakeTuple) ||
      IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    return GetVirtualNodeTarget(node);
  }
#else
  if (IsPrimitiveCNode(node, prim::kPrimDepend) || IsPrimitiveCNode(node, prim::kPrimLoad) ||
      IsPrimitiveCNode(node, prim::kPrimUpdateState) || IsPrimitiveCNode(node, prim::kPrimMakeTuple) ||
      IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    return GetVirtualNodeTarget(node);
  }
#endif
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  return context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
}

std::string GetCNodeTarget(const AnfNodePtr &node) {
  auto kernel_info = node->kernel_info();
  if (kernel_info != nullptr) {
    auto runtime_cache = kernel_info->runtime_cache();
    if (runtime_cache.runtime_cache().is_valid()) {
      auto tmp_target = runtime_cache.runtime_cache().device_target();
      if (!tmp_target.empty()) {
        return tmp_target;
      }
    }
  }

  std::string target;
  auto ori_target = GetOriginNodeTarget(node);
  if (ori_target != kTargetUnDefined) {
    target = ori_target;
  } else {
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  }

  if (kernel_info != nullptr) {
    auto runtime_cache = kernel_info->runtime_cache();
    if (runtime_cache.runtime_cache().is_valid()) {
      runtime_cache.runtime_cache().set_device_target(target);
    }
  }
  return target;
}

bool ContainMultiTarget(const std::vector<AnfNodePtr> &nodes) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string last_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  for (auto &node : nodes) {
    if (node->isa<CNode>()) {
      std::string cur_target = GetCNodeTarget(node);
      if (last_target != cur_target) {
        return true;
      }
      last_target = cur_target;
    }
  }
  return false;
}

bool IsOneOfPrimitive(const AnfNodePtr &node, const PrimitiveSet &prim_set) {
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(node);
  return (prim && prim_set.find(prim) != prim_set.end());
}

bool IsOneOfPrimitiveCNode(const AnfNodePtr &node, const PrimitiveSet &prim_set) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr || cnode->size() == 0) {
    return false;
  }
  return IsOneOfPrimitive(cnode->input(0), prim_set);
}

// Set the sequence nodes' elements use flags to 'new_flag' at specific 'index' position.
void SetSequenceElementsUseFlags(const AbstractBasePtr &abs, std::size_t index, bool new_flag) {
  static const auto enable_eliminate_unused_element = (common::GetEnv("MS_DEV_ENABLE_DDE") != "0");
  if (!enable_eliminate_unused_element) {
    return;
  }

  auto sequence_abs = dyn_cast<abstract::AbstractSequence>(abs);
  if (sequence_abs == nullptr) {
    return;
  }
  if (sequence_abs->sequence_nodes() == nullptr || sequence_abs->sequence_nodes()->empty()) {
    return;
  }
  for (auto &node : *sequence_abs->sequence_nodes()) {
    auto sequence_node = node.lock();
    if (sequence_node == nullptr) {
      MS_LOG(DEBUG) << "The node in sequence_nodes is free.";
      continue;
    }
    auto flags = GetSequenceNodeElementsUseFlags(sequence_node);
    if (flags == nullptr) {
      continue;
    }
    if (index >= flags->size()) {
      MS_LOG(ERROR) << "The index " << index << " is out of range, size is " << flags->size() << ", for "
                    << sequence_node->DebugString();
      return;
    }
    (*flags)[index] = new_flag;
    MS_LOG(DEBUG) << "Set item[" << index << "] use flag as " << new_flag << ", for " << sequence_node->DebugString();
  }
}

// Set the sequence nodes' elements use flags all to 'new_flag'.
void SetSequenceElementsUseFlags(const AbstractBasePtr &abs, bool new_flag) {
  static const auto enable_eliminate_unused_element = (common::GetEnv("MS_DEV_ENABLE_DDE") != "0");
  if (!enable_eliminate_unused_element) {
    return;
  }

  auto sequence_abs = dyn_cast<abstract::AbstractSequence>(abs);
  if (sequence_abs == nullptr) {
    return;
  }
  if (sequence_abs->sequence_nodes() == nullptr || sequence_abs->sequence_nodes()->empty()) {
    return;
  }
  for (auto &weak_node : *sequence_abs->sequence_nodes()) {
    auto sequence_node = weak_node.lock();
    if (sequence_node == nullptr) {
      MS_LOG(DEBUG) << "The node in sequence_nodes is free.";
      continue;
    }
    auto flags = GetSequenceNodeElementsUseFlags(sequence_node);
    if (flags != nullptr) {
      auto &all_flags = (*flags);
      (void)std::transform(all_flags.begin(), all_flags.end(), all_flags.begin(),
                           [&new_flag](bool) -> bool { return new_flag; });
    }
  }
}

// Set the sequence nodes' elements use flags all to 'new_flag' recursively.
void SetSequenceElementsUseFlagsRecursively(const AbstractBasePtr &abs, bool new_flag) {
  static const auto enable_eliminate_unused_element = (common::GetEnv("MS_DEV_ENABLE_DDE") != "0");
  if (!enable_eliminate_unused_element) {
    return;
  }

  SetSequenceElementsUseFlags(abs, new_flag);

  // Check its elements if it's sequence node.
  auto sequence_abs = dyn_cast<abstract::AbstractSequence>(abs);
  if (sequence_abs == nullptr) {
    return;
  }
  for (auto &element : sequence_abs->elements()) {
    SetSequenceElementsUseFlagsRecursively(element, new_flag);
  }
}
}  // namespace mindspore
