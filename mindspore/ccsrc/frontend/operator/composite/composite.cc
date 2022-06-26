/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "frontend/operator/composite/composite.h"
#include <algorithm>

#include "ir/anf.h"
#include "ir/func_graph.h"
#include "abstract/abstract_value.h"
#include "abstract/abstract_function.h"
#include "abstract/dshape.h"
#include "abstract/param_validator.h"
#include "frontend/operator/cc_implementations.h"
#include "frontend/optimizer/opt.h"
#include "utils/symbolic.h"
#include "include/common/pybind_api/api_register.h"
#include "ir/signature.h"
#include "pipeline/jit/debug/trace.h"
#include "utils/ms_context.h"
#include "include/common/utils/utils.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
using AbstractTensor = mindspore::abstract::AbstractTensor;
using FuncGraphAbstractClosure = mindspore::abstract::FuncGraphAbstractClosure;

using mindspore::abstract::AbstractAttribute;
using mindspore::abstract::AbstractBase;
using mindspore::abstract::AbstractDictionary;
using mindspore::abstract::AbstractDictionaryPtr;
using mindspore::abstract::AbstractEllipsis;
using mindspore::abstract::AbstractEllipsisPtr;
using mindspore::abstract::AbstractFunction;
using mindspore::abstract::AbstractFunctionPtr;
using mindspore::abstract::AbstractList;
using mindspore::abstract::AbstractNone;
using mindspore::abstract::AbstractScalar;
using mindspore::abstract::AbstractSlice;
using mindspore::abstract::AbstractTuple;

ElemwiseMap kElemwiseMap = {{"__add__", kPrimScalarAdd}, {"__sub__", kPrimScalarSub}, {"__mul__", kPrimScalarMul},
                            {"__truediv__", nullptr},    {"__floordiv__", nullptr},   {"__mod__", kPrimScalarMod},
                            {"__pow__", kPrimScalarPow}, {"__eq__", kPrimScalarEq},   {"__lt__", kPrimScalarLt},
                            {"__gt__", kPrimScalarGt},   {"__ne__", kPrimScalarNe},   {"__le__", kPrimScalarLe},
                            {"__ge__", kPrimScalarGe}};

ValuePtr kCompositeHyperMap = std::make_shared<HyperMap>();

void HyperMap::Init() {
  if (fn_leaf_) {
    name_ = "hyper_map[" + fn_leaf_->name() + "]";
  }
  signatures_ =
    // def hypermap(func:read, *args:ref):
    std::vector<Signature>({{"func", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault},
                            {"args", SignatureEnumRW::kRWRef, SignatureEnumKind::kKindVarPositional}});
}

HyperMap::HyperMap(bool reverse, const std::shared_ptr<MultitypeFuncGraph> &fn_leaf)
    : MetaFuncGraph("hyper_map"), fn_leaf_(fn_leaf), reverse_(reverse), nonleaf_({kObjectTypeList, kObjectTypeTuple}) {
  Init();
}

HyperMap::HyperMap(const HyperMap &h)
    : MetaFuncGraph("hyper_map"), fn_leaf_(h.fn_leaf_), reverse_(h.reverse_), nonleaf_(h.nonleaf_) {
  Init();
}

AnfNodePtr HyperMap::FullMake(const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg, const ArgsPairList &arg_map) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> inputs;
  if (fn_arg != nullptr) {
    inputs.push_back(fn_arg);
  } else {
    inputs.push_back(NewValueNode(fn_leaf_));
  }

  (void)std::transform(arg_map.begin(), arg_map.end(), std::back_inserter(inputs),
                       [](const std::pair<AnfNodePtr, Any> &item) { return item.first; });
  return func_graph->NewCNodeInOrder(inputs);
}

std::pair<std::string, std::string> HyperMap::GetHyperMapInputIndex(size_t num) {
  std::string error_index;
  std::string next_index;
  const size_t first_index = 1;
  const size_t second_index = 2;
  if (num == first_index) {
    // The first element in HyperMap is func_graph
    error_index = "first";
    next_index = "second";
  } else if (num == second_index) {
    error_index = "second";
    next_index = "third";
  } else {
    error_index = std::to_string(num) + "th";
    next_index = std::to_string(num + 1) + "th";
  }
  return std::pair<std::string, std::string>(error_index, next_index);
}

AnfNodePtr HyperMap::FullMake(const std::shared_ptr<List> &type, const FuncGraphPtr &func_graph,
                              const AnfNodePtr &fn_arg, const ArgsPairList &arg_map) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(type);

  size_t size = type->elements().size();
  size_t num = 0;
  std::ostringstream oss;
  bool is_not_same = false;
  for (auto &item : arg_map) {
    num++;
    auto lhs = std::static_pointer_cast<List>(item.second);
    auto [error_index, next_index] = GetHyperMapInputIndex(num);
    if (lhs == nullptr) {
      MS_LOG(EXCEPTION) << "The " << error_index << " element in HyperMap has wrong type, expected a List, but got "
                        << item.second->ToString() << ".";
    }
    if (lhs->elements().size() != size) {
      oss << "\nThe length of the " << error_index << " element in HyperMap is " << size << ", but the length of the "
          << next_index << " element in HyperMap is " << lhs->elements().size() << ".\n";
      is_not_same = true;
      break;
    }
  }
  if (is_not_same) {
    MS_LOG(EXCEPTION) << "The lists in HyperMap should have the same length. " << oss.str();
  }

  // cannot use shared_from_base() also known as this, as it will make a reference cycle on
  // hypermap and graph generated, it will cause memory leak.
  auto fn_rec = NewValueNode(std::make_shared<HyperMap>(*this));
  constexpr size_t kPrimHoldLen = 1;
  std::vector<AnfNodePtr> inputs;
  inputs.reserve(size + kPrimHoldLen);
  inputs.push_back(NewValueNode(prim::kPrimMakeList));

  for (size_t i = 0; i < size; i++) {
    MS_LOG(DEBUG) << "FullMakeList for the " << i << "th element of the target, reverse_: " << reverse_;
    std::vector<AnfNodePtr> inputs2;
    inputs2.push_back(fn_rec);
    if (fn_arg != nullptr) {
      inputs2.push_back(fn_arg);
    }
    size_t pos = (reverse_ ? (size - 1 - i) : i);
    (void)std::transform(arg_map.begin(), arg_map.end(), std::back_inserter(inputs2),
                         [&func_graph, pos](const std::pair<AnfNodePtr, Any> &item) {
                           return func_graph->NewCNodeInOrder(
                             {NewValueNode(prim::kPrimListGetItem), item.first, NewValueNode(SizeToLong(pos))});
                         });

    auto call_node = func_graph->NewCNodeInOrder(inputs2);
    if (reverse_) {
      inputs.insert(inputs.begin() + 1, call_node);
    } else {
      inputs.emplace_back(call_node);
    }
  }
  return func_graph->NewCNodeInOrder(inputs);
}

AnfNodePtr HyperMap::FullMake(const std::shared_ptr<Tuple> &type, const FuncGraphPtr &func_graph,
                              const AnfNodePtr &fn_arg, const ArgsPairList &arg_map) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(type);

  size_t size = type->elements().size();
  size_t num = 0;
  std::ostringstream oss;
  bool is_not_same = false;
  for (auto &item : arg_map) {
    num++;
    auto lhs = std::static_pointer_cast<Tuple>(item.second);
    auto [error_index, next_index] = GetHyperMapInputIndex(num);
    if (lhs == nullptr) {
      MS_LOG(EXCEPTION) << "The " << error_index << " element in HyperMap has wrong type, expected a Tuple, but got "
                        << item.second->ToString() << ".";
    }
    if (lhs->elements().size() != size) {
      oss << "\nThe length of the " << error_index << " element in HyperMap is " << size << ", but the length of the "
          << next_index << " element in HyperMap is " << lhs->elements().size() << ".\n";
      is_not_same = true;
      break;
    }
  }
  if (is_not_same) {
    MS_LOG(EXCEPTION) << "The length of tuples in HyperMap must be the same. " << oss.str();
  }

  // cannot use shared_from_base() also known as this, as it will make a reference cycle on
  // hypermap and graph generated, it will cause memory leak.
  auto fn_rec = NewValueNode(std::make_shared<HyperMap>(*this));
  constexpr size_t kPrimHoldLen = 1;
  std::vector<AnfNodePtr> inputs;
  inputs.reserve(size + kPrimHoldLen);
  inputs.push_back(NewValueNode(prim::kPrimMakeTuple));

  for (size_t i = 0; i < size; i++) {
    MS_LOG(DEBUG) << "FullMakeTuple for the " << i << "th element of the target, reverse_: " << reverse_;
    std::vector<AnfNodePtr> inputs2;
    inputs2.push_back(fn_rec);
    if (fn_arg != nullptr) {
      inputs2.push_back(fn_arg);
    }
    size_t pos = (reverse_ ? (size - 1 - i) : i);
    (void)std::transform(arg_map.begin(), arg_map.end(), std::back_inserter(inputs2),
                         [&func_graph, &pos](std::pair<AnfNodePtr, Any> item) {
                           return func_graph->NewCNodeInOrder(
                             {NewValueNode(prim::kPrimTupleGetItem), item.first, NewValueNode(SizeToLong(pos))});
                         });

    auto call_node = func_graph->NewCNodeInOrder(inputs2);
    if (reverse_) {
      inputs.insert(inputs.begin() + 1, call_node);
    } else {
      inputs.emplace_back(call_node);
    }
  }

  if (inputs.size() > 1) {
    return func_graph->NewCNodeInOrder(inputs);
  }
  // Empty tuple.
  auto empty_tuple_value = std::make_shared<ValueTuple>(ValuePtrList());
  auto empty_tuple = NewValueNode(empty_tuple_value);
  return empty_tuple;
}

AnfNodePtr HyperMap::Make(const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg, const ArgsPairList &arg_map) {
  bool is_leaf = false;
  TypeId id = kObjectTypeEnd;
  std::pair<AnfNodePtr, TypePtr> pair;
  for (auto &item : arg_map) {
    pair = item;
    id = item.second->type_id();
    // The graph building reaches the leaf situation when there exists type that can not be divided any more.
    if (!nonleaf_.count(id)) {
      is_leaf = true;
      break;
    }
  }

  if (!is_leaf) {
    // In a nonleaf situation, all arguments must have the same generic.
    bool is_not_same = std::any_of(arg_map.begin(), arg_map.end(), [pair](const std::pair<AnfNodePtr, TypePtr> &item) {
      if (item.first != pair.first) {
        return item.second->type_id() != pair.second->type_id();
      }
      return false;
    });
    if (is_not_same) {
      std::ostringstream oss;
      oss << "There are " << arg_map.size() << " inputs of `" << name_ << "`, corresponding type info:\n"
          << trace::GetDebugInfo(func_graph->debug_info()) << "\n";
      int64_t idx = 0;
      std::string str_index = "first";
      const int64_t diff_index = 2;
      for (auto &item : arg_map) {
        // The first element in HyperMap is func_graph
        if (idx == 0) {
          str_index = "second";
        } else if (idx == 1) {
          str_index = "third";
        } else {
          str_index = std::to_string(idx + diff_index) + "th";
        }
        ++idx;
        oss << "The type of the " << str_index << " argument in HyperMap is " << item.second->ToString() << ".\n";
      }
      MS_LOG(EXCEPTION) << "In a nonleaf situation, the types of arguments in HyperMap must be consistent, "
                        << "but the types of arguments are inconsistent.\n"
                        << oss.str();
    }
  }

  switch (id) {
    case kObjectTypeList: {
      auto type = std::static_pointer_cast<List>(pair.second);
      return FullMake(type, func_graph, fn_arg, arg_map);
    }
    case kObjectTypeTuple: {
      auto type = std::static_pointer_cast<Tuple>(pair.second);
      return FullMake(type, func_graph, fn_arg, arg_map);
    }
    default:
      return FullMake(func_graph, fn_arg, arg_map);
  }
}

FuncGraphPtr HyperMap::GenerateFromTypes(const TypePtrList &args_spec_list) {
  FuncGraphPtr ptr_graph = std::make_shared<FuncGraph>();
  ptr_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  ptr_graph->set_flag(FUNC_GRAPH_FLAG_SPECIALIZE_PARAMETER, true);
  ptr_graph->debug_info()->set_name("hyper_map");

  AnfNodePtr ptrFnArg = nullptr;
  std::size_t i = 0;
  ArgsPairList argmap;
  if (fn_leaf_ == nullptr) {
    ptrFnArg = ptr_graph->add_parameter();
    i = 1;
  }

  std::size_t size = args_spec_list.size();
  for (; i < size; ++i) {
    argmap.push_back(std::make_pair(ptr_graph->add_parameter(), args_spec_list[i]));
  }

  ptr_graph->set_output(Make(ptr_graph, ptrFnArg, argmap));
  return ptr_graph;
}

abstract::AbstractBasePtrList HyperMap::NormalizeArgs(const AbstractBasePtrList &args_spec_list) const {
  if (fn_leaf_ == nullptr) {
    if (args_spec_list.empty()) {
      MS_LOG(EXCEPTION) << "The size of arguments in list should not be empty. But the size of arguments is 0.";
    }
    MS_EXCEPTION_IF_NULL(args_spec_list[0]);
    // Assert that hypermap's function param does not contain free variables
    if (args_spec_list[0]->isa<FuncGraphAbstractClosure>()) {
      auto graph_func = dyn_cast<FuncGraphAbstractClosure>(args_spec_list[0]);
      auto func_graph = graph_func->func_graph();
      if (func_graph->parent() != nullptr) {
        MS_LOG(EXCEPTION) << "HyperMap don't support Closure with free variable yet.";
      }
    }
  }

  AbstractBasePtrList broadened;
  (void)std::transform(args_spec_list.begin(), args_spec_list.end(), std::back_inserter(broadened),
                       [](const AbstractBasePtr &arg) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(arg);
                         return arg->Broaden();
                       });
  return broadened;
}

REGISTER_PYBIND_DEFINE(HyperMap_, ([](const py::module *m) {
                         (void)py::class_<HyperMapPy, MetaFuncGraph, std::shared_ptr<HyperMapPy>>(*m, "HyperMap_")
                           .def(py::init<bool, std::shared_ptr<MultitypeFuncGraph>>(), py::arg("reverse"),
                                py::arg("ops"))
                           .def(py::init<bool>(), py::arg("reverse"));
                       }));

bool CheckSequenceAllTensor(const abstract::AbstractTuplePtr &tuple) {
  MS_EXCEPTION_IF_NULL(tuple);
  for (size_t i = 0; i < tuple->size(); ++i) {
    if (!(*tuple)[i]->isa<abstract::AbstractUndetermined>() &&
        !((*tuple)[i]->isa<abstract::AbstractTuple>() &&
          CheckSequenceAllTensor((*tuple)[i]->cast<abstract::AbstractTuplePtr>()))) {
      return false;
    }
  }
  return true;
}

bool CheckTailGradFristSequence(const abstract::AbstractSequencePtr &sequeue, bool enable_tuple_grad) {
  MS_EXCEPTION_IF_NULL(sequeue);
  return sequeue->size() > 1 && (*sequeue)[1] != nullptr &&
         ((*sequeue)[1]->isa<abstract::AbstractUndetermined>() ||
          (MsContext::GetInstance()->get_param<bool>(MS_CTX_GRAD_FOR_SCALAR) && (*sequeue)[1]->BuildType() != nullptr &&
           (*sequeue)[1]->BuildType()->isa<Number>()) ||
          ((*sequeue)[1]->isa<abstract::AbstractTuple>() && enable_tuple_grad &&
           CheckSequenceAllTensor((*sequeue)[1]->cast<abstract::AbstractTuplePtr>())));
}

namespace {
void GenerateSequenceFuncGraphByPosition(const FuncGraphPtr &res, const abstract::AbstractSequencePtr &sequeue,
                                         const abstract::AbstractSequencePtr &pos, bool enable_tuple_grad) {
  if (pos == nullptr) {
    MS_LOG(EXCEPTION) << "Return grad by position, but the grad_position is empty!";
  }
  AnfNodePtr tuple_parameter = res->add_parameter();
  std::vector<AnfNodePtr> pos_elements;
  PrimitivePtr pos_op = nullptr;
  if (pos->isa<AbstractTuple>()) {
    pos_elements.push_back(NewValueNode(prim::kPrimMakeTuple));
    pos_op = prim::kPrimTupleGetItem;
  } else {
    pos_elements.push_back(NewValueNode(prim::kPrimMakeList));
    pos_op = prim::kPrimListGetItem;
  }
  AnfNodePtr pos_value = nullptr;
  AnfNodePtr pos_value_adjust = nullptr;
  auto pos_parameter = res->add_parameter();
  if (pos->size() == 1) {
    pos_value = res->NewCNode({NewValueNode(pos_op), pos_parameter, NewValueNode(SizeToLong(0))});
    pos_value_adjust = res->NewCNode({NewValueNode(prim::kPrimScalarAdd), pos_value, NewValueNode(SizeToLong(1))});
    if (CheckTailGradFristSequence(sequeue, enable_tuple_grad)) {
      res->set_output(res->NewCNode({NewValueNode(pos_op), tuple_parameter, pos_value_adjust}));
    } else {
      res->set_output(NewValueNode(std::make_shared<ValueTuple>(std::vector<ValuePtr>{})));
    }
  } else {
    for (size_t i = 0; i < pos->size(); ++i) {
      pos_value = res->NewCNode({NewValueNode(pos_op), pos_parameter, NewValueNode(SizeToLong(i))});
      pos_value_adjust = res->NewCNode({NewValueNode(prim::kPrimScalarAdd), pos_value, NewValueNode(SizeToLong(1))});
      pos_elements.push_back(res->NewCNodeInOrder({NewValueNode(pos_op), tuple_parameter, pos_value_adjust}));
    }
    if (pos_elements.size() > 1) {
      res->set_output(res->NewCNodeInOrder(pos_elements));
    } else if (pos->isa<AbstractTuple>()) {  // Empty tuple.
      auto empty_tuple_value = std::make_shared<ValueTuple>(ValuePtrList());
      auto empty_tuple = NewValueNode(empty_tuple_value);
      res->set_output(empty_tuple);
    } else {  // Empty list.
      auto empty_list_value = std::make_shared<ValueList>(ValuePtrList());
      auto empty_list = NewValueNode(empty_list_value);
      res->set_output(empty_list);
    }
  }
}
}  // namespace

FuncGraphPtr Tail::GenerateSequenceFuncGraph(const abstract::AbstractSequencePtr &sequeue,
                                             const abstract::AbstractSequencePtr &pos) const {
  MS_EXCEPTION_IF_NULL(sequeue);

  FuncGraphPtr res = std::make_shared<FuncGraph>();
  res->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  res->debug_info()->set_name("tail");

  if (tail_type_ == kGradFirst) {
    AnfNodePtr tuple_parameter = res->add_parameter();
    PrimitivePtr getitem_op = nullptr;
    if (sequeue->isa<AbstractTuple>()) {
      getitem_op = prim::kPrimTupleGetItem;
    } else {
      getitem_op = prim::kPrimListGetItem;
    }
    if (CheckTailGradFristSequence(sequeue, enable_tuple_grad_)) {
      res->set_output(res->NewCNode({NewValueNode(getitem_op), tuple_parameter, NewValueNode(SizeToLong(1))}));
    } else {
      res->set_output(NewValueNode(std::make_shared<ValueTuple>(ValuePtrList())));
    }
    return res;
  }

  if (tail_type_ == kGradByPosition) {
    GenerateSequenceFuncGraphByPosition(res, sequeue, pos, enable_tuple_grad_);
    return res;
  }

  AnfNodePtr tuple_parameter = res->add_parameter();
  std::vector<AnfNodePtr> elements;
  PrimitivePtr op = nullptr;
  if (sequeue->isa<AbstractTuple>()) {
    elements.push_back(NewValueNode(prim::kPrimMakeTuple));
    op = prim::kPrimTupleGetItem;
  } else {
    elements.push_back(NewValueNode(prim::kPrimMakeList));
    op = prim::kPrimListGetItem;
  }
  for (size_t i = 1; i < sequeue->size(); ++i) {
    if (tail_type_ == kGradAll) {
      MS_EXCEPTION_IF_NULL((*sequeue)[i]);
      if ((*sequeue)[i]->isa<abstract::AbstractUndetermined>() ||
          (MsContext::GetInstance()->get_param<bool>(MS_CTX_GRAD_FOR_SCALAR) && (*sequeue)[i]->BuildType() != nullptr &&
           (*sequeue)[i]->BuildType()->isa<Number>())) {
        elements.push_back(res->NewCNodeInOrder({NewValueNode(op), tuple_parameter, NewValueNode(SizeToLong(i))}));
      }
    } else {
      elements.push_back(res->NewCNodeInOrder({NewValueNode(op), tuple_parameter, NewValueNode(SizeToLong(i))}));
    }
  }
  if (elements.size() > 1) {
    res->set_output(res->NewCNodeInOrder(elements));
    return res;
  } else if (sequeue->isa<AbstractTuple>()) {  // Empty tuple.
    auto empty_tuple_value = std::make_shared<ValueTuple>(ValuePtrList());
    auto empty_tuple = NewValueNode(empty_tuple_value);
    res->set_output(empty_tuple);
    return res;
  } else {  // Empty list.
    auto empty_list_value = std::make_shared<ValueList>(ValuePtrList());
    auto empty_list = NewValueNode(empty_list_value);
    res->set_output(empty_list);
    return res;
  }
}

FuncGraphPtr Tail::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  if (args_spec_list.size() < 1) {
    MS_LOG(EXCEPTION) << "Tail requires a non-empty tuple.";
  }

  AbstractBasePtr a = args_spec_list[0];
  if (a->isa<AbstractTuple>() || a->isa<AbstractList>()) {
    if (args_spec_list.size() > 1) {
      AbstractBasePtr pos = args_spec_list[1];
      if (pos->isa<AbstractTuple>() || pos->isa<AbstractList>()) {
        return GenerateSequenceFuncGraph(a->cast<abstract::AbstractSequencePtr>(),
                                         pos->cast<abstract::AbstractSequencePtr>());
      }
      MS_LOG(EXCEPTION) << "'Tail' arg1 must be AbstractTuple or AbstractList, but got " << pos->ToString();
    }
    return GenerateSequenceFuncGraph(a->cast<abstract::AbstractSequencePtr>());
  }

  MS_LOG(EXCEPTION) << "'Tail' arg0 must be AbstractTuple or AbstractList, but got " << a->ToString();
}

REGISTER_PYBIND_DEFINE(
  Tail_, ([](const py::module *m) {
    (void)py::class_<Tail, MetaFuncGraph, std::shared_ptr<Tail>>(*m, "Tail_").def(py::init<std::string &>());
  }));

FuncGraphPtr MakeTupleGradient::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  int64_t tuple_size = SizeToLong(args_spec_list.size());

  std::ostringstream ss;
  ss << "▶make_tuple_" << tuple_size;
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  fg->debug_info()->set_name(ss.str());

  std::vector<AnfNodePtr> params;
  params.push_back(NewValueNode(prim::kPrimMakeTuple));
  for (int64_t i = 0; i < tuple_size; ++i) {
    params.push_back(fg->add_parameter());
  }

  // make fprob first result, maketuple's forward result.
  AnfNodePtr out = fg->NewCNodeInOrder(params);

  // make fprob second result, maketuple's backward function.
  FuncGraphPtr b = std::make_shared<FuncGraph>();

  ss.str(std::string());
  ss.clear();
  ss << "◀make_tuple_" << tuple_size;
  b->debug_info()->set_name(ss.str());
  AnfNodePtr dout = b->add_parameter();

  std::vector<AnfNodePtr> grads;
  grads.push_back(NewValueNode(prim::kPrimMakeTuple));
  grads.push_back(NewEnviron(b));
  for (int64_t i = 0; i < tuple_size; ++i) {
    grads.push_back(b->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), dout, NewValueNode(i)}));
  }

  b->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  b->set_output(b->NewCNodeInOrder(grads));

  fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  fg->set_output(fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), out, NewValueNode(b)}));
  (void)fg->transforms().emplace("primal", FuncGraphTransform(prim::kPrimMakeTuple));
  return fg;
}

FuncGraphPtr MakeListGradient::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  int64_t list_size = SizeToLong(args_spec_list.size());

  std::ostringstream ss;
  ss << "▶make_list_" << list_size;
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  fg->debug_info()->set_name(ss.str());

  std::vector<AnfNodePtr> params;
  params.push_back(NewValueNode(prim::kPrimMakeList));
  for (int64_t i = 0; i < list_size; ++i) {
    params.push_back(fg->add_parameter());
  }

  // make fprob first result, maketuple's forward result.
  AnfNodePtr out = fg->NewCNodeInOrder(params);

  // make fprob second result, maketuple's backward function.
  FuncGraphPtr b = std::make_shared<FuncGraph>();

  ss.str(std::string());
  ss.clear();
  ss << "◀make_list_" << list_size;
  b->debug_info()->set_name(ss.str());
  AnfNodePtr dout = b->add_parameter();

  std::vector<AnfNodePtr> grads;
  grads.push_back(NewValueNode(prim::kPrimMakeTuple));
  grads.push_back(NewEnviron(b));
  for (int64_t i = 0; i < list_size; ++i) {
    grads.push_back(b->NewCNodeInOrder({NewValueNode(prim::kPrimListGetItem), dout, NewValueNode(i)}));
  }

  b->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  b->set_output(b->NewCNodeInOrder(grads));

  fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  fg->set_output(fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), out, NewValueNode(b)}));
  (void)fg->transforms().emplace("primal", FuncGraphTransform(prim::kPrimMakeList));
  return fg;
}

GradOperation::GradOperation(const std::string &name, bool get_all, bool get_by_list, bool sens_param,
                             bool get_by_position)
    : MetaFuncGraph(name),
      get_all_(get_all),
      get_by_list_(get_by_list),
      sens_param_(sens_param),
      get_by_position_(get_by_position) {
  if (get_by_position) {
    signatures_ =
      // def grad(func:read, weight_list:ref, position_list:ref):
      std::vector<Signature>({{"func", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault},
                              {"weight_list", SignatureEnumRW::kRWRef, SignatureEnumKind::kKindDefault},
                              {"position_list", SignatureEnumRW::kRWRef, SignatureEnumKind::kKindDefault}});
  } else if (get_by_list) {
    signatures_ =
      // def grad(func:read, weight_list:ref):
      std::vector<Signature>({{"func", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault},
                              {"weight_list", SignatureEnumRW::kRWRef, SignatureEnumKind::kKindDefault}});
  }
}

FuncGraphPtr GradOperation::GetGrad(const AnfNodePtr &j, const AnfNodePtr &weights, const AnfNodePtr &position,
                                    const std::vector<AnfNodePtr> &forward_graph_params, bool enable_tuple_grad,
                                    const std::vector<AnfNodePtr> &weight_args) {
  FuncGraphPtr k_child = std::make_shared<FuncGraph>();
  k_child->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  k_child->set_flag(FUNC_GRAPH_FLAG_K_GRAPH, true);

  AnfNodePtr weights_node = nullptr;
  AnfNodePtr position_node = nullptr;
  if (weights != nullptr) {
    weights_node = weights;
  } else if (!weight_args.empty()) {
    weights_node = k_child->NewCNodeInOrder(weight_args);
  }
  if (position != nullptr) {
    position_node = position;
  }

  std::vector<AnfNodePtr> inputs;
  inputs.push_back(j);
  for (size_t i = 0; i < forward_graph_params.size(); ++i) {
    inputs.push_back(k_child->add_parameter());
  }
  auto k_app = k_child->NewCNodeInOrder(inputs);

  auto tuple_get_item = NewValueNode(prim::kPrimTupleGetItem);
  auto f_app = k_child->NewCNodeInOrder({tuple_get_item, k_app, NewValueNode(static_cast<int64_t>(0))});
  auto bprop = k_child->NewCNodeInOrder({tuple_get_item, k_app, NewValueNode(static_cast<int64_t>(1))});

  GradByParameter(k_child, f_app, bprop, weights_node, position_node, enable_tuple_grad);
  return k_child;
}

// Do grad by the parameter of GradOperation.
void GradOperation::GradByParameter(const FuncGraphPtr &k_child, const AnfNodePtr &f_app, const AnfNodePtr &bprop,
                                    const AnfNodePtr &weights, const AnfNodePtr &position, bool enable_tuple_grad) {
  MS_EXCEPTION_IF_NULL(k_child);

  AnfNodePtr bprop_arg = nullptr;
  if (sens_param_) {
    bprop_arg = k_child->add_parameter();
  } else {
    auto ones_like = prim::GetPythonOps("ones_like");
    bprop_arg = k_child->NewCNodeInOrder({NewValueNode(ones_like), f_app});
  }

  AnfNodePtr b_app = k_child->NewCNodeInOrder({bprop, bprop_arg});

  CNodePtr fv_bprop = nullptr;
  if (get_by_list_) {
    // python code: grads = hyper_map(F.partial(env_get, env), weights)
    AnfNodePtr env =
      k_child->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), b_app, NewValueNode(static_cast<int64_t>(0))});
    AnfNodePtr partial_env_get =
      k_child->NewCNodeInOrder({NewValueNode(prim::kPrimPartial), NewValueNode(prim::GetPythonOps("env_get")), env});
    MetaFuncGraphPtr hyper_map = std::make_shared<HyperMap>();
    fv_bprop = k_child->NewCNodeInOrder({NewValueNode(hyper_map), partial_env_get, weights});
  }

  CNodePtr inputs_bprop = nullptr;
  if (get_by_position_) {
    TailPtr tail_grad_by_position = std::make_shared<Tail>("tail_grad_by_position", kGradByPosition);
    inputs_bprop = k_child->NewCNodeInOrder({NewValueNode(tail_grad_by_position), b_app, position});
    k_child->set_output(inputs_bprop);
    return;
  }
  if (get_all_) {
    TailPtr tail_grad_all = std::make_shared<Tail>("tail_grad_all", kGradAll);
    inputs_bprop = k_child->NewCNodeInOrder({NewValueNode(tail_grad_all), b_app});
  }

  // Gradients wrt inputs and parameters
  if (fv_bprop != nullptr && inputs_bprop != nullptr) {
    k_child->set_output(k_child->NewCNodeInOrder({NewValueNode(kPrimMakeTuple), inputs_bprop, fv_bprop}));
    return;
  }

  // Gradients wrt parameters
  if (fv_bprop != nullptr) {
    k_child->set_output(fv_bprop);
    return;
  }

  // Gradients wrt inputs
  if (inputs_bprop != nullptr) {
    k_child->set_output(inputs_bprop);
    return;
  }
  // Gradients wrt first input.
  // b_app returns (EnvInstance(grads wrt params), grads wrt input0, grads wrt input1, ...),
  // so obtain first input grad by setting tail_type of Tail to kGradFirst.
  TailPtr tail_grad_first = std::make_shared<Tail>("tail_grad_first", kGradFirst);
  tail_grad_first->set_enable_tuple_grad(enable_tuple_grad);
  k_child->set_output(k_child->NewCNodeInOrder({NewValueNode(tail_grad_first), b_app}));
}

// Generate the graph.
FuncGraphPtr GradOperation::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  if (args_spec_list.empty()) {
    MS_LOG(EXCEPTION)
      << "'GradOperation' requires a forward network or function as an input, while the input is empty.";
  }

  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  AbstractFunctionPtr fn = dyn_cast<AbstractFunction>(args_spec_list[0]);
  if (fn == nullptr) {
    MS_LOG(EXCEPTION) << "For 'GradOperation', the first argument must be a 'Function' or 'Cell', but got "
                      << args_spec_list[0]->ToString();
  }

  // Waiting for implementation.
  auto real_fn = dyn_cast<FuncGraphAbstractClosure>(fn);
  MS_EXCEPTION_IF_NULL(real_fn);

  FuncGraphPtr forward_graph = real_fn->func_graph();
  MS_EXCEPTION_IF_NULL(forward_graph);
  forward_graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
  FuncGraphPtr grad_fg = nullptr;
  {
    TraceGuard g(std::make_shared<TraceGradOperation>(forward_graph->debug_info()));
    grad_fg = std::make_shared<FuncGraph>();
  }
  auto nparam = forward_graph->parameters().size();

  std::ostringstream ss;
  ss << "grad{" << nparam << "}";
  grad_fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  grad_fg->debug_info()->set_name(ss.str());
  ParameterPtr param_graph = grad_fg->add_parameter();

  AnfNodePtr weights = nullptr;
  if (get_by_list_) {
    weights = grad_fg->add_parameter();
  }
  AnfNodePtr position = nullptr;
  if (get_by_position_) {
    weights = grad_fg->add_parameter();
    position = grad_fg->add_parameter();
  }

  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimJ));
  inputs.push_back(param_graph);
  auto j = grad_fg->NewCNodeInOrder(inputs);
  // df is checked in GetGrad
  FuncGraphPtr k_child = nullptr;
  {
    TraceGuard guard(std::make_shared<TraceGradOperation>(forward_graph->debug_info()));
    k_child = GetGrad(j, weights, position, forward_graph->parameters(), forward_graph->has_flag("enable_tuple_grad"));
  }
  grad_fg->set_output(NewValueNode(k_child));

  return grad_fg;
}

REGISTER_PYBIND_DEFINE(GradOperation_, ([](const py::module *m) {
                         (void)py::class_<GradOperation, MetaFuncGraph, std::shared_ptr<GradOperation>>(
                           *m, "GradOperation_")
                           .def(py::init<std::string &>(), py::arg("fn"))
                           .def(py::init<std::string &, bool, bool, bool, bool>(), py::arg("fn"), py::arg("get_all"),
                                py::arg("get_by_list"), py::arg("sens_param"), py::arg("get_by_position"));
                       }));

// Generate the vmap_graph.
VmapOperation::VmapOperation(const std::string &name) : MetaFuncGraph(name) {
  auto default_zero = std::make_shared<Int64Imm>(static_cast<int64_t>(0));
  signatures_ =
    // def vmap(func:read, in_axes:ref, out_axes:ref):
    std::vector<Signature>({{"func", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault},
                            {"in_axes", SignatureEnumRW::kRWRef, SignatureEnumKind::kKindDefault, default_zero,
                             SignatureEnumDType::kDTypeEmptyDefaultValue},
                            {"out_axes", SignatureEnumRW::kRWRef, SignatureEnumKind::kKindDefault, default_zero,
                             SignatureEnumDType::kDTypeEmptyDefaultValue}});
}

FuncGraphPtr VmapOperation::GetVmap(const AnfNodePtr &vmap, const std::vector<AnfNodePtr> &forward_graph_params) {
  FuncGraphPtr vmap_child = std::make_shared<FuncGraph>();
  vmap_child->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  vmap_child->set_flag(FUNC_GRAPH_FLAG_K_GRAPH, true);

  std::vector<AnfNodePtr> inputs;
  inputs.push_back(vmap);
  for (size_t i = 0; i < forward_graph_params.size(); ++i) {
    inputs.push_back(vmap_child->add_parameter());
  }
  auto vmap_app = vmap_child->NewCNodeInOrder(inputs);
  vmap_child->set_output(vmap_app);

  return vmap_child;
}

namespace {
bool IsAxesAllNone(const ValuePtr &axes) {
  MS_EXCEPTION_IF_NULL(axes);
  ValueSequencePtr axes_seq = dyn_cast<ValueSequence>(axes);
  auto axes_seq_value = axes_seq->value();
  if (std::all_of(axes_seq_value.begin(), axes_seq_value.end(), [](const ValuePtr &axes_value_ptr) {
        if (axes_value_ptr->isa<ValueSequence>()) {
          return IsAxesAllNone(axes_value_ptr);
        }
        if (!axes_value_ptr->isa<None>()) {
          return false;
        }
        return true;
      })) {
    return true;
  }
  return false;
}

ValuePtr CheckAxes(const AbstractBasePtr &axes_abs, const bool &is_in_axes = false, int nparam = 0) {
  ValuePtr axes_value = nullptr;
  auto axes_name = is_in_axes ? "in_axes" : "out_axes";

  auto axes_abs_sequence = dyn_cast<abstract::AbstractSequence>(axes_abs);
  if (axes_abs_sequence != nullptr) {
    axes_value = axes_abs->cast<abstract::AbstractSequencePtr>()->ElementsBuildValue<ValueTuple>();
    MS_EXCEPTION_IF_NULL(axes_value);
    if (is_in_axes) {
      ValueSequencePtr in_axes_seq = dyn_cast<ValueSequence>(axes_value);
      int in_axes_size = SizeToInt(in_axes_seq->size());
      if (nparam != in_axes_size) {
        MS_LOG(EXCEPTION) << "When vmap`s '" << axes_name
                          << "' is a tuple or list, and its size must be equal to the number of arguments of 'fn': "
                          << nparam << ", but got size: " << in_axes_size << ".";
      }
    }
    bool elem_all_none = IsAxesAllNone(axes_value);
    if (elem_all_none) {
      MS_LOG(EXCEPTION) << "The '" << axes_name << "' of 'vmap' cannot be all None, but got " << axes_value->ToString()
                        << ".";
    }
  } else {
    axes_value = axes_abs->BuildValue();
    MS_EXCEPTION_IF_NULL(axes_value);
    if (axes_value->isa<None>()) {
      MS_LOG(EXCEPTION) << "The '" << axes_name << "' of 'vmap' cannot be a single None.";
    } else if (!axes_value->isa<Int64Imm>()) {
      MS_LOG(EXCEPTION) << "The axis in vmap`s '" << axes_name << "' can only be of type Int or None, but got "
                        << axes_abs->ToString() << ".";
    }
  }
  return axes_value;
}
}  // namespace

FuncGraphPtr VmapOperation::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  if (args_spec_list.empty()) {
    MS_LOG(EXCEPTION) << "'VmapOperation' requires a network or function as an input, while the input is empty.";
  }

  constexpr auto kVmapOperationInputNum = 3;
  const std::string op_name = "vmap";
  CheckArgsSize(op_name, args_spec_list, kVmapOperationInputNum);

  auto fn_arg = args_spec_list[0];
  auto in_axes_arg = args_spec_list[1];
  auto out_axes_arg = args_spec_list[2];

  AbstractFunctionPtr fn = dyn_cast<AbstractFunction>(fn_arg);
  if (fn == nullptr) {
    MS_LOG(EXCEPTION) << "'VmapOperation' arg0 must be a 'Function' or 'Cell', but got " << fn_arg->ToString() << ".";
  }

  auto real_fn = dyn_cast<FuncGraphAbstractClosure>(fn);
  if (real_fn == nullptr) {
    MS_LOG(EXCEPTION) << "'VmapOperation' arg0 " << fn->ToString() << " cast to 'FuncGraphAbstractClosure' failed.";
  }

  FuncGraphPtr orig_graph = real_fn->func_graph();
  MS_EXCEPTION_IF_NULL(orig_graph);
  orig_graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
  FuncGraphPtr vmap_fg = nullptr;
  {
    TraceGuard guard(std::make_shared<TraceVmapOperation>(orig_graph->debug_info()));
    vmap_fg = std::make_shared<FuncGraph>();
  }
  int nparam = SizeToInt(orig_graph->parameters().size());

  std::ostringstream ss;
  ss << "vmap{" << nparam << "}";
  vmap_fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  vmap_fg->debug_info()->set_name(ss.str());

  // add parameter for `fn`, `in_axes` and `out_axes` respectively.
  ParameterPtr param_graph = vmap_fg->add_parameter();
  (void)vmap_fg->add_parameter();
  (void)vmap_fg->add_parameter();

  // Validity verification of in_axes and out_axes
  ValuePtr in_axes = CheckAxes(in_axes_arg, true, nparam);
  ValuePtr out_axes = CheckAxes(out_axes_arg);

  PrimitivePtr kprim_vmap = std::make_shared<Primitive>(prim::kVmap, kSideEffectPropagate);
  kprim_vmap->set_attr("in_axes", in_axes);
  kprim_vmap->set_attr("out_axes", out_axes);

  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(kprim_vmap));
  inputs.push_back(param_graph);
  auto vmap = vmap_fg->NewCNodeInOrder(inputs);

  FuncGraphPtr vmap_child = nullptr;
  {
    TraceGuard guard(std::make_shared<TraceVmapOperation>(orig_graph->debug_info()));
    vmap_child = GetVmap(vmap, orig_graph->parameters());
  }

  vmap_fg->set_output(NewValueNode(vmap_child));
  return vmap_fg;
}

REGISTER_PYBIND_DEFINE(VmapOperation_, ([](const py::module *m) {
                         (void)py::class_<VmapOperation, MetaFuncGraph, std::shared_ptr<VmapOperation>>(
                           *m, "VmapOperation_")
                           .def(py::init<const std::string &>(), py::arg("fn"));
                       }));

TaylorOperation::TaylorOperation(const std::string &name) : MetaFuncGraph(name) {
  // def Taylor(func:read):
  signatures_ = std::vector<Signature>({{"func", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault}});
}

FuncGraphPtr TaylorOperation::GetTaylorGrad(const AnfNodePtr &k, const std::vector<AnfNodePtr> &forward_graph_params) {
  FuncGraphPtr k_child = std::make_shared<FuncGraph>();
  k_child->set_flag(FUNC_GRAPH_FLAG_CORE, true);

  std::vector<AnfNodePtr> inputs;
  inputs.push_back(k);
  MS_LOG(INFO) << "TaylorOperation forward input size " << forward_graph_params.size();
  for (size_t i = 0; i < forward_graph_params.size(); ++i) {
    inputs.push_back(k_child->add_parameter());
  }
  // Taylor(fn)(input params)
  auto k_app = k_child->NewCNodeInOrder(inputs);

  k_child->set_output(k_app);
  return k_child;
}

// Generate the graph to calculate higher order derivatives.
FuncGraphPtr TaylorOperation::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  if (args_spec_list.empty()) {
    MS_LOG(EXCEPTION)
      << "'TaylorOperation' requires a forward network or function as an input, while the input is empty.";
  }

  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  AbstractFunctionPtr fn = dyn_cast<AbstractFunction>(args_spec_list[0]);
  if (fn == nullptr) {
    MS_LOG(EXCEPTION) << "'TaylorOperation' arg0 must be a 'Function' or 'Cell', but got "
                      << args_spec_list[0]->ToString();
  }

  auto real_fn = dyn_cast<FuncGraphAbstractClosure>(fn);
  MS_EXCEPTION_IF_NULL(real_fn);

  FuncGraphPtr forward_graph = real_fn->func_graph();
  MS_EXCEPTION_IF_NULL(forward_graph);
  forward_graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
  FuncGraphPtr grad_fg = nullptr;
  MS_LOG(INFO) << "'TaylorOperation' forward_graph" << forward_graph->debug_info();
  grad_fg = std::make_shared<FuncGraph>();
  auto nparam = forward_graph->parameters().size();

  std::ostringstream ss;
  ss << "taylorgrad{" << nparam << "}";
  grad_fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  grad_fg->debug_info()->set_name(ss.str());
  ParameterPtr param_graph = grad_fg->add_parameter();

  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimTaylor));
  inputs.push_back(param_graph);
  // Taylor(fn)
  auto mark_taylor = grad_fg->NewCNodeInOrder(inputs);
  FuncGraphPtr k_child = nullptr;
  {
    TraceGuard guard(std::make_shared<TraceGradOperation>(forward_graph->debug_info()));
    k_child = GetTaylorGrad(mark_taylor, forward_graph->parameters());
  }
  grad_fg->set_output(NewValueNode(k_child));
  // return Taylor(fn)(inputs)
  return grad_fg;
}

REGISTER_PYBIND_DEFINE(TaylorOperation_, ([](const py::module *m) {
                         (void)py::class_<TaylorOperation, MetaFuncGraph, std::shared_ptr<TaylorOperation>>(
                           *m, "TaylorOperation_")
                           .def(py::init<const std::string &>(), py::arg("fn"));
                       }));

// Generate the ListMap func graph.
FuncGraphPtr ListMap::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  size_t args_num = args_spec_list.size();
  // args: fn, list1, list2, ...
  if (args_num < 2) {
    MS_LOG(EXCEPTION) << "The list_map operator must need at least two arguments, but the size of arguments is "
                      << args_num << ".";
  }

  for (size_t i = 1; i < args_num; ++i) {
    if (typeid(args_spec_list[i]) != typeid(AbstractBase)) {
      // The function currently not be use
      MS_LOG(EXCEPTION) << "The type of arguments of list_map operator must be lists. But got "
                        << args_spec_list[i]->ToString();
    }
  }

  FuncGraphPtr fg_ptr = std::make_shared<FuncGraph>();
  fg_ptr->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  fg_ptr->debug_info()->set_name("list_map");
  AnfNodePtr fn = fg_ptr->add_parameter();

  std::vector<AnfNodePtr> lists;
  for (size_t i = 1; i < args_num; ++i) {
    lists.push_back(fg_ptr->add_parameter());
  }

  std::vector<AnfNodePtr> iters;
  (void)std::transform(lists.begin(), lists.end(), std::back_inserter(iters), [fg_ptr](AnfNodePtr item) {
    return fg_ptr->NewCNodeInOrder({NewValueNode(std::string("list_iter")), item});
  });

  std::vector<AnfNodePtr> nexts;
  (void)std::transform(iters.begin(), iters.end(), std::back_inserter(nexts), [fg_ptr](AnfNodePtr item) {
    return fg_ptr->NewCNodeInOrder({NewValueNode(std::string("next")), item});
  });

  std::vector<AnfNodePtr> values;
  (void)std::transform(nexts.begin(), nexts.end(), std::back_inserter(values), [fg_ptr](AnfNodePtr item) {
    return fg_ptr->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), item});
  });

  (void)std::transform(nexts.begin(), nexts.end(), std::back_inserter(iters), [fg_ptr](AnfNodePtr item) {
    return fg_ptr->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleGetItem), item, NewValueNode(static_cast<int64_t>(1))});
  });

  (void)values.insert(values.begin(), fn);
  AnfNodePtr cnode_graph = fg_ptr->NewCNodeInOrder(values);
  AnfNodePtr resl = fg_ptr->NewCNodeInOrder({NewValueNode(prim::kPrimMakeList), cnode_graph});

  FuncGraphPtr fgnext_ptr = std::make_shared<FuncGraph>();
  fgnext_ptr->debug_info()->set_name("body");

  FuncGraphPtr fgcond_ptr = std::make_shared<FuncGraph>();
  fgcond_ptr->debug_info()->set_name("cond");

  MakeCond(lists, fgnext_ptr, fgcond_ptr);
  MakeNext(lists, fgcond_ptr, fgnext_ptr);

  CNodePtr output_cnode = fg_ptr->NewCNodeInOrder({NewValueNode(fgcond_ptr), fn, resl});

  auto inputs = output_cnode->inputs();
  (void)inputs.insert(inputs.end(), iters.begin(), iters.end());
  output_cnode->set_inputs(inputs);

  fg_ptr->set_output(output_cnode);
  return fg_ptr;
}

void ListMap::MakeCond(const std::vector<AnfNodePtr> &lists, const FuncGraphPtr &fgnext_ptr,
                       const FuncGraphPtr &fg_ptr) {
  MS_EXCEPTION_IF_NULL(fg_ptr);

  AnfNodePtr fn = fg_ptr->add_parameter();
  AnfNodePtr resl = fg_ptr->add_parameter();

  std::vector<AnfNodePtr> iters;
  (void)std::transform(lists.begin(), lists.end(), std::back_inserter(iters),
                       [fg_ptr](AnfNodePtr) { return fg_ptr->add_parameter(); });

  std::vector<AnfNodePtr> hasnexts;
  (void)std::transform(iters.begin(), iters.end(), std::back_inserter(hasnexts), [fg_ptr](AnfNodePtr item) {
    return fg_ptr->NewCNodeInOrder({NewValueNode(std::string("hasnext")), item});
  });

  // cond = reduce(lambda a, b: g.apply(P.bool_and, a, b), hasnexts)
  FuncGraphPtr fgtrue_ptr = std::make_shared<FuncGraph>();
  fgtrue_ptr->debug_info()->set_name("ftrue");
  fgtrue_ptr->set_flag(FUNC_GRAPH_FLAG_CORE, true);

  CNodePtr fgtrue_output_cnode = fgtrue_ptr->NewCNodeInOrder({NewValueNode(fgnext_ptr), fn, resl});
  auto inputs = fgtrue_output_cnode->inputs();
  (void)inputs.insert(inputs.end(), iters.begin(), iters.end());
  fgtrue_output_cnode->set_inputs(inputs);
  fgtrue_ptr->set_output(fgtrue_output_cnode);

  FuncGraphPtr fgfalse_ptr = std::make_shared<FuncGraph>();
  fgfalse_ptr->debug_info()->set_name("ffalse");
  fgfalse_ptr->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  fgfalse_ptr->set_output(resl);

  AnfNodePtr output_cnode = fg_ptr->NewCNodeInOrder({NewValueNode(prim::kPrimSwitch), NewValueNode(std::string("cond")),
                                                     NewValueNode(fgtrue_ptr), NewValueNode(fgfalse_ptr)});
  fgtrue_ptr->set_output(output_cnode);
}

void ListMap::MakeNext(const std::vector<AnfNodePtr> &lists, const FuncGraphPtr &fgcond_ptr,
                       const FuncGraphPtr &fg_ptr) {
  MS_EXCEPTION_IF_NULL(fg_ptr);
  AnfNodePtr fn = fg_ptr->add_parameter();

  std::vector<AnfNodePtr> iters;
  (void)std::transform(lists.begin(), lists.end(), std::back_inserter(iters),
                       [fg_ptr](AnfNodePtr) { return fg_ptr->add_parameter(); });

  std::vector<AnfNodePtr> nexts;
  (void)std::transform(iters.begin(), iters.end(), std::back_inserter(nexts), [fg_ptr](AnfNodePtr item) {
    return fg_ptr->NewCNodeInOrder({NewValueNode(std::string("next")), item});
  });

  std::vector<AnfNodePtr> values;
  (void)std::transform(nexts.begin(), nexts.end(), std::back_inserter(values), [fg_ptr](AnfNodePtr item) {
    return fg_ptr->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), item, nullptr});
  });

  iters.clear();
  (void)std::transform(nexts.begin(), nexts.end(), std::back_inserter(iters), [fg_ptr](AnfNodePtr item) {
    return fg_ptr->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleGetItem), item, NewValueNode(static_cast<int64_t>(1))});
  });

  (void)values.insert(values.begin(), fn);
  AnfNodePtr cnode_graph = fg_ptr->NewCNodeInOrder(values);
  AnfNodePtr resl = fg_ptr->NewCNodeInOrder({NewValueNode(prim::kPrimListAppend), cnode_graph});
  CNodePtr output_cnode = fg_ptr->NewCNodeInOrder({NewValueNode(fgcond_ptr), fn, resl});

  auto inputs = output_cnode->inputs();
  (void)inputs.insert(inputs.end(), iters.begin(), iters.end());
  output_cnode->set_inputs(inputs);
  fg_ptr->set_output(output_cnode);
}

FuncGraphPtr TupleAdd::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  // args: tuple1, tuple2
  abstract::CheckArgsSize("TupleAdd", args_spec_list, 2);
  AbstractBasePtr abs_a = args_spec_list[0];
  AbstractBasePtr abs_b = args_spec_list[1];

  abstract::AbstractTuplePtr a_tuple = dyn_cast<AbstractTuple>(abs_a);
  abstract::AbstractTuplePtr b_tuple = dyn_cast<AbstractTuple>(abs_b);
  if (a_tuple == nullptr || b_tuple == nullptr) {
    TypePtrList types;
    (void)std::transform(args_spec_list.begin(), args_spec_list.end(), std::back_inserter(types),
                         [](const AbstractBasePtr &arg) -> TypePtr {
                           MS_EXCEPTION_IF_NULL(arg);
                           return arg->BuildType();
                         });
    auto stub = GenerateStubFunc(types);
    if (stub != nullptr) {
      MS_LOG(DEBUG) << "GenerateStubFunc for TupleAdd "
                    << ", function: " << stub->ToString();
      return stub;
    }
    MS_LOG(EXCEPTION) << "The type of argument in TupleAdd operator should be tuple, but the first argument is "
                      << args_spec_list[0]->ToString() << ", the second argument is" << args_spec_list[1]->ToString();
  }

  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  AnfNodePtr p_tup_a = ret->add_parameter();
  AnfNodePtr p_tup_b = ret->add_parameter();

  std::vector<AnfNodePtr> elems;
  elems.push_back(NewValueNode(prim::kPrimMakeTuple));

  int64_t tuple_size = SizeToLong(a_tuple->size());
  for (int64_t i = 0; i < tuple_size; ++i) {
    elems.push_back(ret->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), p_tup_a, NewValueNode(i)}));
  }

  tuple_size = SizeToLong(b_tuple->size());
  for (int64_t i = 0; i < tuple_size; ++i) {
    elems.push_back(ret->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), p_tup_b, NewValueNode(i)}));
  }

  ret->set_output(ret->NewCNodeInOrder(elems));
  return ret;
}

int64_t GetArgScalarValue(const abstract::AbstractScalarPtr &scalar, const std::string &) {
  MS_EXCEPTION_IF_NULL(scalar);
  return GetValue<int64_t>(scalar->BuildValue());
}

int64_t GetPositiveIndex(int64_t index, int64_t length) {
  if (index < 0) {
    index += length;
  }
  return index;
}

int64_t CheckSliceMember(const AbstractBasePtr &member, int64_t default_value, const std::string &member_name) {
  MS_EXCEPTION_IF_NULL(member);

  if (member->isa<AbstractScalar>()) {
    return GetArgScalarValue(dyn_cast<AbstractScalar>(member), member_name);
  }

  if (member->isa<AbstractNone>()) {
    return default_value;
  }

  if (member->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError)
      << "The argument of SliceMember operator must be a Scalar or None or constant Tensor, but got a variable Tensor";
  }
  MS_EXCEPTION(TypeError)
    << "The argument of SliceMember operator must be a Scalar or None or constant Tensor, but got "
    << member->BuildType()->ToString();
}

void GenerateTupleSliceParameter(const abstract::AbstractSequencePtr &tuple, const AbstractSlicePtr &slice,
                                 int64_t *start_index, int64_t *stop_index, int64_t *step_value) {
  MS_EXCEPTION_IF_NULL(tuple);
  MS_EXCEPTION_IF_NULL(slice);
  MS_EXCEPTION_IF_NULL(start_index);
  MS_EXCEPTION_IF_NULL(stop_index);
  MS_EXCEPTION_IF_NULL(step_value);

  const std::string start_name("Slice start index");
  const std::string stop_name("Slice stop index");
  const std::string step_name("Slice step value");

  int64_t tuple_size = SizeToLong(tuple->size());
  int64_t start_default = 0;
  int64_t stop_default = tuple_size;
  int64_t step_default = 1;

  *step_value = CheckSliceMember(slice->step(), step_default, step_name);
  if (*step_value == 0) {
    MS_EXCEPTION(ValueError) << "Slice step cannot be zero.";
  }

  if (*step_value < 0) {
    start_default = tuple_size - 1;
    stop_default = -1;
  }

  *start_index = CheckSliceMember(slice->start(), start_default, start_name);
  *stop_index = CheckSliceMember(slice->stop(), stop_default, stop_name);

  if (*start_index < -tuple_size) {
    *start_index = 0;
  }
  if (*stop_index > tuple_size) {
    *stop_index = tuple_size;
  }
  if (*start_index > tuple_size || *stop_index < -tuple_size) {
    *start_index = 0;
    *stop_index = 0;
  }

  *start_index = GetPositiveIndex(*start_index, tuple_size);
  if (!slice->stop()->isa<AbstractNone>()) {
    *stop_index = GetPositiveIndex(*stop_index, tuple_size);
  }
}

FuncGraphPtr SequenceSlice::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  // slice a tuple
  // args: tuple, start index, end index, step
  auto seq_pair = CheckArgs(args_spec_list);
  auto sequence = seq_pair.first;
  auto slice = seq_pair.second;
  int64_t start_index;
  int64_t stop_index;
  int64_t step_value;
  GenerateTupleSliceParameter(sequence, slice, &start_index, &stop_index, &step_value);

  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  AnfNodePtr p_tuple = ret->add_parameter();
  (void)ret->add_parameter();

  std::vector<AnfNodePtr> elems;
  elems.push_back(NewValueNode(prim_));
  if (step_value > 0) {
    for (int64_t index = start_index; index < stop_index; index = index + step_value) {
      elems.push_back(ret->NewCNodeInOrder({NewValueNode(get_item_), p_tuple, NewValueNode(index)}));
    }
  } else {
    for (int64_t index = start_index; index > stop_index; index = index + step_value) {
      elems.push_back(ret->NewCNodeInOrder({NewValueNode(get_item_), p_tuple, NewValueNode(index)}));
    }
  }

  ret->set_output(ret->NewCNodeInOrder(elems));
  return ret;
}

std::pair<abstract::AbstractSequencePtr, abstract::AbstractSlicePtr> TupleSlice::CheckArgs(
  const AbstractBasePtrList &args_spec_list) {
  constexpr size_t arg_size = 2;
  abstract::CheckArgsSize("TupleSlice", args_spec_list, arg_size);
  auto sequence = abstract::CheckArg<abstract::AbstractSequence>("TupleSlice", args_spec_list, 0);
  AbstractSlicePtr slice = abstract::CheckArg<AbstractSlice>("TupleSlice", args_spec_list, 1);
  return std::make_pair(sequence, slice);
}

std::pair<abstract::AbstractSequencePtr, abstract::AbstractSlicePtr> ListSlice::CheckArgs(
  const AbstractBasePtrList &args_spec_list) {
  constexpr size_t arg_size = 2;
  abstract::CheckArgsSize("ListSlice", args_spec_list, arg_size);
  auto sequence = abstract::CheckArg<abstract::AbstractSequence>("ListSlice", args_spec_list, 0);
  AbstractSlicePtr slice = abstract::CheckArg<AbstractSlice>("ListSlice", args_spec_list, 1);
  return std::make_pair(sequence, slice);
}

FuncGraphPtr TupleGetItemTensor::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  // select indexed item
  // args: tuple of items, index
  const std::string op_name = std::string("TupleGetItemTensor");
  const size_t inputs_size = 2;
  abstract::CheckArgsSize(op_name, args_spec_list, inputs_size);
  auto ret_graph = std::make_shared<FuncGraph>();
  ret_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  auto functions = ret_graph->add_parameter();
  auto index = ret_graph->add_parameter();

  ret_graph->set_output(ret_graph->NewCNodeInOrder({NewValueNode(prim::kPrimSwitchLayer), index, functions}));
  return ret_graph;
}

REGISTER_PYBIND_DEFINE(TupleAdd_, ([](const py::module *m) {
                         (void)py::class_<TupleAdd, MetaFuncGraph, std::shared_ptr<TupleAdd>>(*m, "TupleAdd_")
                           .def(py::init<std::string &>());
                       }));

REGISTER_PYBIND_DEFINE(TupleSlice_, ([](const py::module *m) {
                         (void)py::class_<TupleSlice, MetaFuncGraph, std::shared_ptr<TupleSlice>>(*m, "TupleSlice_")
                           .def(py::init<std::string &>());
                       }));

REGISTER_PYBIND_DEFINE(TupleGetItemTensor_, ([](const py::module *m) {
                         (void)py::class_<TupleGetItemTensor, MetaFuncGraph, std::shared_ptr<TupleGetItemTensor>>(
                           *m, "TupleGetItemTensor_")
                           .def(py::init<std::string &>());
                       }));

REGISTER_PYBIND_DEFINE(ListSlice_, ([](const py::module *m) {
                         (void)py::class_<ListSlice, MetaFuncGraph, std::shared_ptr<ListSlice>>(*m, "ListSlice_")
                           .def(py::init<std::string &>());
                       }));

namespace {
FuncGraphPtr GetShard(const AnfNodePtr &shard, const std::vector<AnfNodePtr> &origin_graph_params) {
  FuncGraphPtr shard_child = std::make_shared<FuncGraph>();
  shard_child->set_flag(FUNC_GRAPH_FLAG_CORE, true);

  std::vector<AnfNodePtr> inputs;
  inputs.reserve(origin_graph_params.size() + 1);
  (void)inputs.emplace_back(shard);
  for (size_t i = 0; i < origin_graph_params.size(); ++i) {
    (void)inputs.emplace_back(shard_child->add_parameter());
  }
  auto shard_app = shard_child->NewCNodeInOrder(std::move(inputs));

  shard_child->set_output(shard_app);
  return shard_child;
}
}  // namespace

FuncGraphPtr Shard::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  constexpr size_t shard_input_size = 5;
  if (args_spec_list.size() != shard_input_size) {
    MS_LOG(EXCEPTION) << "'Shard' requires " << shard_input_size
                      << " inputs. Includes a Cell or function, in_axes, out_axes, device and level.";
  }

  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  AbstractFunctionPtr fn = dyn_cast<AbstractFunction>(args_spec_list[0]);
  if (fn == nullptr) {
    MS_LOG(EXCEPTION) << "'Shard' arg0 must be a 'Function' or 'Cell', but got " << args_spec_list[0]->ToString()
                      << ".";
  }

  auto real_fn = dyn_cast<FuncGraphAbstractClosure>(fn);
  MS_EXCEPTION_IF_NULL(real_fn);
  FuncGraphPtr origin_graph = real_fn->func_graph();
  MS_EXCEPTION_IF_NULL(origin_graph);
  origin_graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
  FuncGraphPtr shard_fg = nullptr;
  {
    TraceGuard g(std::make_shared<TraceShard>(origin_graph->debug_info()));
    shard_fg = std::make_shared<FuncGraph>();
  }
  // Create the debug info
  auto parameter_size = origin_graph->parameters().size();
  std::ostringstream ss;
  ss << "shard{" << parameter_size << "}";
  shard_fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  shard_fg->debug_info()->set_name(ss.str());
  // Make the Shard node.
  std::vector<AnfNodePtr> inputs;
  inputs.reserve(args_spec_list.size() + 1);
  (void)inputs.emplace_back(NewValueNode(prim::kPrimShard));
  for (size_t i = 0; i < args_spec_list.size(); ++i) {
    (void)inputs.emplace_back(shard_fg->add_parameter());
  }
  auto shard = shard_fg->NewCNodeInOrder(std::move(inputs));

  FuncGraphPtr shard_child = nullptr;
  {
    TraceGuard guard(std::make_shared<TraceShard>(shard_fg->debug_info()));
    shard_child = GetShard(shard, origin_graph->parameters());
  }
  shard_fg->set_output(NewValueNode(shard_child));
  return shard_fg;
}

REGISTER_PYBIND_DEFINE(Shard_, ([](const py::module *m) {
                         (void)py::class_<Shard, MetaFuncGraph, std::shared_ptr<Shard>>(*m, "Shard_")
                           .def(py::init<const std::string &>(), py::arg("fn"));
                       }));
}  // namespace prim
}  // namespace mindspore
