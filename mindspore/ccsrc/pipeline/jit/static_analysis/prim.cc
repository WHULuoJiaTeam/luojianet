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

#include "pipeline/jit/static_analysis/prim.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <string>
#include <utility>

#include "ir/anf.h"
#include "utils/hash_set.h"
#include "frontend/operator/cc_implementations.h"
#include "frontend/operator/ops.h"
#include "frontend/operator/composite/do_signature.h"
#include "frontend/operator/prim_to_function.h"
#include "abstract/utils.h"
#include "utils/log_adapter.h"
#include "utils/symbolic.h"
#include "pipeline/jit/resource.h"
#include "pipeline/jit/parse/resolve.h"
#include "pipeline/jit/pipeline.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
#include "pipeline/jit/debug/trace.h"
#include "include/common/utils/convert_utils.h"
#include "include/common/utils/convert_utils_py.h"
#include "utils/ms_context.h"
#include "pipeline/jit/parse/data_converter.h"
#include "abstract/primitive_infer_map.h"
#include "abstract/param_validator.h"
#include "utils/ms_utils.h"
#include "utils/shape_utils.h"
#include "utils/parallel_node_check.h"
#include "frontend/operator/ops_front_infer_function.h"

namespace mindspore {
namespace abstract {
using mindspore::parse::PyObjectWrapper;

mindspore::HashSet<std::string> prims_to_skip_undetermined_infer{
  prim::kPrimMakeTuple->name(),  prim::kPrimMakeList->name(),   prim::kPrimSwitch->name(),
  prim::kPrimEnvironSet->name(), prim::kPrimEnvironGet->name(), prim::kPrimLoad->name(),
  prim::kPrimUpdateState->name()};

// The Python primitives who visit tuple/list elements, but not consume all elements.
// Including:
// - Consume no element. For instance, MakeTuple.
// - Consume partial elements, not all. For instance, TupleGetItem.
// Map{"primitive name", {vector<int>:"index to transparent pass, -1 means all elements"}}
mindspore::HashMap<std::string, std::vector<int>> prims_transparent_pass_sequence{
  {prim::kPrimReturn->name(), std::vector({0})},       {prim::kPrimDepend->name(), std::vector({0})},
  {prim::kPrimIdentity->name(), std::vector({0})},     {prim::kPrimMakeTuple->name(), std::vector({-1})},
  {prim::kPrimMakeList->name(), std::vector({-1})},    {prim::kPrimListAppend->name(), std::vector({0})},
  {prim::kPrimTupleGetItem->name(), std::vector({0})}, {prim::kPrimListGetItem->name(), std::vector({0})}};

EvalResultPtr DoSignatureEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                        const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(engine);
  MS_EXCEPTION_IF_NULL(out_conf);
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &config) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(config);
                         MS_EXCEPTION_IF_NULL(config->ObtainEvalResult());
                         return config->ObtainEvalResult()->abstract();
                       });

  // Do undetermined infer firstly.
  auto do_signature = prim_->cast<prim::DoSignaturePrimitivePtr>();
  MS_EXCEPTION_IF_NULL(do_signature);
  auto &func = do_signature->function();
  auto do_signature_func = dyn_cast<Primitive>(func);
  if (do_signature_func != nullptr) {
    if (prims_to_skip_undetermined_infer.find(do_signature_func->name()) == prims_to_skip_undetermined_infer.end()) {
      auto ret_abstract = AbstractEval(args_spec_list);
      if (ret_abstract != nullptr) {
        MS_LOG(DEBUG) << "DoSignatureEvaluator eval Undetermined for " << do_signature_func->name()
                      << ", ret_abstract: " << ret_abstract->ToString();
        return ret_abstract;
      }
    }
  }

  // Create new CNode with old CNode.
  if (out_conf->node() == nullptr || !out_conf->node()->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Node of out_conf should be CNode";
  }
  auto out_cnode = dyn_cast<CNode>(out_conf->node());
  MS_EXCEPTION_IF_NULL(out_cnode);
  const auto &out_node_inputs = out_cnode->inputs();
  if (out_cnode->inputs().size() == 0 || (out_node_inputs.size() - 1) != args_conf_list.size()) {
    MS_LOG(EXCEPTION) << "Op: " << func->ToString() << " args size should equal to inputs size minus 1, but args size "
                      << args_conf_list.size() << ", inputs size " << out_node_inputs.size();
  }
  AnfNodePtrList args_inputs{out_node_inputs.begin() + 1, out_node_inputs.end()};
  AnfNodePtr new_node = nullptr;
  ScopePtr scope = out_conf->node()->scope();
  ScopeGuard scope_guard(scope);
  if (bound_node() != nullptr) {
    TraceGuard trace_guard(std::make_shared<TraceDoSignature>(bound_node()->debug_info()));
    new_node = prim::GenerateCNode(out_cnode->func_graph(), prim_->ToString(), func, args_spec_list, args_inputs);
  } else {
    new_node = prim::GenerateCNode(out_cnode->func_graph(), prim_->ToString(), func, args_spec_list, args_inputs);
  }
  // Update new CNode info.
  auto new_cnode = dyn_cast<CNode>(new_node);
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->CloneCNodeInfo(out_cnode);

  // Do forward with old config and new config.
  AnfNodeConfigPtr new_conf = engine->MakeConfig(new_node, out_conf->context(), out_conf->func_graph());
  return engine->ForwardConfig(out_conf, new_conf);
}

static AbstractBasePtrList GetUnpackGraphSpecArgsList(AbstractBasePtrList args_spec_list, bool need_unpack) {
  // arg[0] is the func graph to unpack, ignore it
  AbstractBasePtrList specialize_args_before_unpack(args_spec_list.begin() + 1, args_spec_list.end());
  AbstractBasePtrList graph_specialize_args;
  if (need_unpack) {
    for (size_t index = 0; index < specialize_args_before_unpack.size(); index++) {
      MS_EXCEPTION_IF_NULL(specialize_args_before_unpack[index]);
      if (specialize_args_before_unpack[index]->isa<AbstractTuple>()) {
        auto arg_tuple = specialize_args_before_unpack[index]->cast<AbstractTuplePtr>();
        std::transform(arg_tuple->elements().begin(), arg_tuple->elements().end(),
                       std::back_inserter(graph_specialize_args), [](AbstractBasePtr abs) { return abs; });
      } else if (specialize_args_before_unpack[index]->isa<AbstractDictionary>()) {
        auto arg_dict = specialize_args_before_unpack[index]->cast<AbstractDictionaryPtr>();
        auto dict_elems = arg_dict->elements();
        (void)std::transform(
          dict_elems.begin(), dict_elems.end(), std::back_inserter(graph_specialize_args),
          [](const AbstractAttribute &item) { return std::make_shared<AbstractKeywordArg>(item.first, item.second); });
      } else {
        MS_LOG(EXCEPTION) << "UnpackGraph require args should be tuple or dict, but got "
                          << specialize_args_before_unpack[index]->ToString();
      }
    }
  } else {
    graph_specialize_args = specialize_args_before_unpack;
  }
  return graph_specialize_args;
}

EvalResultPtr UnpackGraphEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                        const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(engine);
  MS_EXCEPTION_IF_NULL(out_conf);
  MS_EXCEPTION_IF_NULL(out_conf->node());
  if (out_conf->node() == nullptr || !out_conf->node()->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Node of out_conf should be CNode";
  }

  auto unpack_graph = prim_->cast<prim::UnpackGraphPrimitivePtr>();
  MS_EXCEPTION_IF_NULL(unpack_graph);
  auto out_node = out_conf->node()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(out_node);
  const auto &out_node_inputs = out_node->inputs();
  if (out_node->inputs().empty() || (out_node_inputs.size() - 1) != args_conf_list.size()) {
    MS_LOG(EXCEPTION) << "UnpackGraphPrimitive"
                      << " args size should equal to inputs size minus 1, but args size " << args_conf_list.size()
                      << ", inputs size " << out_node_inputs.size();
  }
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &ref) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(ref);
                         MS_EXCEPTION_IF_NULL(ref->ObtainEvalResult());
                         return ref->ObtainEvalResult()->abstract();
                       });
  // get the forward graph
  if (args_spec_list.empty()) {
    MS_LOG(EXCEPTION) << "args_spec_list can't be empty.";
  }
  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  auto fn = args_spec_list[0]->cast<AbstractFunctionPtr>();
  if (fn == nullptr) {
    MS_LOG(EXCEPTION) << "UnpackGraphPrimitive arg0 must be AbstractFunction, but " << args_spec_list[0]->ToString();
  }
  auto real_fn = fn->cast<FuncGraphAbstractClosurePtr>();
  MS_EXCEPTION_IF_NULL(real_fn);
  FuncGraphPtr forward_graph = real_fn->func_graph();
  MS_EXCEPTION_IF_NULL(forward_graph);
  AbstractBasePtrList graph_specialize_args =
    GetUnpackGraphSpecArgsList(args_spec_list, unpack_graph->need_unpack_args());
  AbstractBasePtrList graph_specialize_args_without_sens;
  if (unpack_graph->with_sens_in_args() && graph_specialize_args.empty()) {
    MS_EXCEPTION(ValueError) << "Grad with sens, but the sens is not provided.";
  }
  (void)std::transform(graph_specialize_args.begin(),
                       graph_specialize_args.end() - (unpack_graph->with_sens_in_args() ? 1 : 0),
                       std::back_inserter(graph_specialize_args_without_sens), [](AbstractBasePtr abs) { return abs; });
  auto new_graph = forward_graph->GenerateGraph(graph_specialize_args_without_sens);
  engine->func_graph_manager()->AddFuncGraph(new_graph);
  ScopePtr scope = kDefaultScope;
  if (out_conf != nullptr) {
    scope = out_conf->node()->scope();
  }
  ScopeGuard scope_guard(scope);
  AnfNodePtr new_vnode = NewValueNode(new_graph);
  AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_vnode, out_conf->context(), out_conf->func_graph());

  return engine->ForwardConfig(out_conf, fn_conf);
}

AnfNodePtr MixedPrecisionCastHelper(const AnfNodePtr &source_node, const AbstractBasePtr &node_type,
                                    const AnfNodePtr &target_type, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(node_type);
  MS_EXCEPTION_IF_NULL(func_graph);
  AnfNodePtr target_node = source_node;
  if (node_type->isa<AbstractTensor>()) {
    auto x = node_type->cast<AbstractTensorPtr>();
    if (x->element()->BuildType()->isa<Float>()) {
      auto cast = prim::GetPythonOps("cast", "mindspore.ops.functional");
      MS_EXCEPTION_IF_NULL(cast);
      target_node = func_graph->NewCNodeAfter(source_node, {NewValueNode(cast), source_node, target_type});
    }
  } else if (node_type->isa<AbstractTuple>()) {
    auto x = node_type->cast<AbstractTuplePtr>();
    auto &items = x->elements();
    std::vector<AnfNodePtr> nodes;
    nodes.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    int64_t idx = 0;
    for (const auto &item : items) {
      AnfNodePtr tuple_node =
        func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), source_node, NewValueNode(idx)});
      AnfNodePtr node = MixedPrecisionCastHelper(tuple_node, item, target_type, func_graph);
      nodes.emplace_back(node);
      ++idx;
    }
    target_node = func_graph->NewCNode(nodes);
  } else if (node_type->isa<AbstractDictionary>()) {
    auto x = node_type->cast<AbstractDictionaryPtr>();
    auto &items = x->elements();
    std::vector<AnfNodePtr> dict_key_nodes;
    std::vector<AnfNodePtr> dict_value_nodes;
    dict_key_nodes.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    dict_value_nodes.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    for (const auto &item : items) {
      AnfNodePtr dict_value_node =
        func_graph->NewCNode({NewValueNode(prim::kPrimDictGetItem), source_node, NewValueNode(item.first)});
      AnfNodePtr node = MixedPrecisionCastHelper(dict_value_node, item.second, target_type, func_graph);
      dict_key_nodes.emplace_back(NewValueNode(item.first));
      dict_value_nodes.emplace_back(node);
    }
    target_node =
      func_graph->NewCNode({NewValueNode(prim::kPrimMakeDict), func_graph->NewCNode(std::move(dict_key_nodes)),
                            func_graph->NewCNode(std::move(dict_value_nodes))});
  } else if (node_type->isa<AbstractKeywordArg>()) {
    auto x = node_type->cast<AbstractKeywordArgPtr>();
    std::string kwarg_key = x->get_key();
    AnfNodePtr kwarg_value_node =
      func_graph->NewCNode({NewValueNode(prim::kPrimExtractKeywordArg), NewValueNode(kwarg_key), source_node});
    AnfNodePtr node = MixedPrecisionCastHelper(kwarg_value_node, x->get_arg(), target_type, func_graph);
    target_node = func_graph->NewCNode({NewValueNode(prim::kPrimMakeKeywordArg), NewValueNode(kwarg_key), node});
  }
  return target_node;
}

EvalResultPtr MixedPrecisionCastEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                               const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(engine);
  AbstractBasePtrList args_spec_list;
  MS_EXCEPTION_IF_NULL(out_conf);
  if (out_conf->node() == nullptr || !out_conf->node()->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Node of out_conf should be CNode";
  }
  auto out_node = out_conf->node()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(out_node);
  const auto &out_node_inputs = out_node->inputs();
  if (out_node->inputs().empty() || (out_node_inputs.size() - 1) != args_conf_list.size()) {
    MS_LOG(EXCEPTION) << "MixedPrecisionCast"
                      << " args size should equal to inputs size minus 1, but args size " << args_conf_list.size()
                      << ", inputs size " << out_node_inputs.size();
  }
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &ref) -> AbstractBasePtr { return ref->ObtainEvalResult()->abstract(); });

  ScopeGuard scope_guard(out_conf->node()->scope());
  TraceGuard trace_guard(std::make_shared<TraceMixedPrecision>(out_conf->node()->debug_info()));

  FuncGraphPtr func_graph = out_node->func_graph();
  constexpr size_t source_node_index = 2;
  if (out_node_inputs.size() <= source_node_index) {
    MS_LOG(EXCEPTION) << "Input size:" << out_node_inputs.size() << " should bigger than 2.";
  }

  AnfNodePtr new_node =
    MixedPrecisionCastHelper(out_node_inputs[source_node_index], args_spec_list[1], out_node_inputs[1], func_graph);
  AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_node, out_conf->context(), out_conf->func_graph());

  if (new_node->isa<CNode>()) {
    auto new_cnode = new_node->cast<CNodePtr>();
    new_cnode->CloneCNodeInfo(out_node);
  }
  return engine->ForwardConfig(out_conf, fn_conf);
}

namespace {
py::object BuildValue(const ValuePtr &value_ptr) {
  if (value_ptr == nullptr) {
    return py::none();
  } else {
    return ValueToPyData(value_ptr);
  }
}

py::object AbstractTupleValueToPython(const AbstractTuplePtr &tuple_abs) {
  MS_EXCEPTION_IF_NULL(tuple_abs);
  auto value = tuple_abs->BuildValue();
  if (value->isa<AnyValue>()) {
    return py::none();
  }
  const auto &elements = tuple_abs->elements();
  size_t len = elements.size();
  py::tuple value_tuple(len);
  for (size_t i = 0; i < len; ++i) {
    value_tuple[i] = ConvertAbstractToPython(elements[i], true)[ATTR_VALUE];
  }
  return std::move(value_tuple);
}

py::dict AbstractTupleToPython(const AbstractBasePtr &abs_base, bool only_convert_value) {
  auto arg_tuple = dyn_cast<AbstractTuple>(abs_base);
  MS_EXCEPTION_IF_NULL(arg_tuple);
  auto dic = py::dict();
  if (only_convert_value) {
    dic[ATTR_VALUE] = AbstractTupleValueToPython(arg_tuple);
    return dic;
  }
  size_t len = arg_tuple->size();
  py::tuple shape_tuple(len);
  py::tuple dtype_tuple(len);
  py::tuple value_tuple(len);
  py::tuple min_value_tuple(len);
  py::tuple max_value_tuple(len);
  py::tuple min_shape_tuple(len);
  py::tuple max_shape_tuple(len);
  bool dyn_shape = false;
  bool dyn_value = false;

  for (size_t i = 0; i < len; i++) {
    py::dict out = ConvertAbstractToPython(arg_tuple->elements()[i]);
    shape_tuple[i] = out[ATTR_SHAPE];
    dtype_tuple[i] = out[ATTR_DTYPE];
    value_tuple[i] = out[ATTR_VALUE];

    // Elements in tuple is tensor shape value.
    if (out.contains(py::str(ATTR_MIN_VALUE)) && out.contains(py::str(ATTR_MAX_VALUE))) {
      min_value_tuple[i] = out[ATTR_MIN_VALUE];
      max_value_tuple[i] = out[ATTR_MAX_VALUE];
      dyn_value = true;
    } else {
      min_value_tuple[i] = out[ATTR_VALUE];
      max_value_tuple[i] = out[ATTR_VALUE];
    }

    // Elements in tuple is tensor, which shape is dynamic.
    if (out.contains(py::str(ATTR_MIN_SHAPE)) && out.contains(py::str(ATTR_MAX_SHAPE))) {
      min_shape_tuple[i] = out[ATTR_MIN_SHAPE];
      max_shape_tuple[i] = out[ATTR_MAX_SHAPE];
      dyn_shape = true;
    } else {
      min_shape_tuple[i] = out[ATTR_SHAPE];
      max_shape_tuple[i] = out[ATTR_SHAPE];
    }
  }
  dic[ATTR_SHAPE] = shape_tuple;
  dic[ATTR_DTYPE] = dtype_tuple;
  MS_EXCEPTION_IF_NULL(arg_tuple->BuildValue());
  if (arg_tuple->BuildValue()->isa<AnyValue>()) {
    dic[ATTR_VALUE] = py::none();
  } else {
    dic[ATTR_VALUE] = value_tuple;
  }

  if (dyn_value) {
    dic[ATTR_MIN_VALUE] = min_value_tuple;
    dic[ATTR_MAX_VALUE] = max_value_tuple;
  }
  if (dyn_shape) {
    dic[ATTR_MIN_SHAPE] = min_shape_tuple;
    dic[ATTR_MAX_SHAPE] = max_shape_tuple;
  }

  return dic;
}

py::object AbstractListValueToPython(const AbstractListPtr &list_abs) {
  MS_EXCEPTION_IF_NULL(list_abs);
  auto value = list_abs->BuildValue();
  if (value->isa<AnyValue>()) {
    return py::none();
  }
  const auto &elements = list_abs->elements();
  size_t len = elements.size();
  py::list value_list(len);
  for (size_t i = 0; i < len; ++i) {
    value_list[i] = ConvertAbstractToPython(elements[i], true)[ATTR_VALUE];
  }
  return std::move(value_list);
}

py::dict AbstractListToPython(const AbstractBasePtr &abs_base, bool only_convert_value) {
  auto arg_list = dyn_cast<AbstractList>(abs_base);
  MS_EXCEPTION_IF_NULL(arg_list);
  auto dic = py::dict();
  if (only_convert_value) {
    dic[ATTR_VALUE] = AbstractListValueToPython(arg_list);
    return dic;
  }
  size_t len = arg_list->size();
  py::list shape_list(len);
  py::list dtype_list(len);
  py::list value_list(len);
  py::list min_value_list(len);
  py::list max_value_list(len);
  py::list min_shape_list(len);
  py::list max_shape_list(len);
  bool dyn_value = false;
  bool dyn_shape = false;

  for (size_t i = 0; i < len; i++) {
    py::dict out = ConvertAbstractToPython(arg_list->elements()[i]);
    shape_list[i] = out[ATTR_SHAPE];
    dtype_list[i] = out[ATTR_DTYPE];
    value_list[i] = out[ATTR_VALUE];

    // Elements in list is tensor, which value is dynamic.
    if (out.contains(py::str(ATTR_MIN_VALUE)) && out.contains(py::str(ATTR_MAX_VALUE))) {
      min_value_list[i] = out[ATTR_MIN_VALUE];
      max_value_list[i] = out[ATTR_MAX_VALUE];
      dyn_value = true;
    } else {
      min_value_list[i] = out[ATTR_VALUE];
      max_value_list[i] = out[ATTR_VALUE];
    }

    // Elements in list is tensor, which shape is dynamic.
    if (out.contains(py::str(ATTR_MIN_SHAPE)) && out.contains(py::str(ATTR_MAX_SHAPE))) {
      min_shape_list[i] = out[ATTR_MIN_SHAPE];
      max_shape_list[i] = out[ATTR_MAX_SHAPE];
      dyn_shape = true;
    } else {
      min_shape_list[i] = out[ATTR_SHAPE];
      max_shape_list[i] = out[ATTR_SHAPE];
    }
  }

  dic[ATTR_SHAPE] = shape_list;
  dic[ATTR_DTYPE] = dtype_list;
  MS_EXCEPTION_IF_NULL(arg_list->BuildValue());
  if (arg_list->BuildValue()->isa<AnyValue>()) {
    dic[ATTR_VALUE] = py::none();
  } else {
    dic[ATTR_VALUE] = value_list;
  }

  if (dyn_value) {
    dic[ATTR_MIN_VALUE] = min_value_list;
    dic[ATTR_MAX_VALUE] = max_value_list;
  }
  if (dyn_shape) {
    dic[ATTR_MIN_SHAPE] = min_shape_list;
    dic[ATTR_MAX_SHAPE] = max_shape_list;
  }

  return dic;
}

void ConvertAbstractTensorToPython(const AbstractBasePtr &abs_base, bool only_convert_value, py::dict *dic) {
  auto arg_tensor = dyn_cast<AbstractTensor>(abs_base);
  MS_EXCEPTION_IF_NULL(dic);
  MS_EXCEPTION_IF_NULL(arg_tensor);
  if (only_convert_value) {
    (*dic)[ATTR_VALUE] = BuildValue(arg_tensor->BuildValue());
    return;
  }
  MS_EXCEPTION_IF_NULL(arg_tensor->shape());
  (*dic)[ATTR_SHAPE] = arg_tensor->shape()->shape();
  const auto &min_shape = arg_tensor->shape()->min_shape();
  const auto &max_shape = arg_tensor->shape()->max_shape();
  if (!min_shape.empty() && !max_shape.empty()) {
    (*dic)[ATTR_MIN_SHAPE] = min_shape;
    (*dic)[ATTR_MAX_SHAPE] = max_shape;
  }

  auto min_value = arg_tensor->get_min_value();
  auto max_value = arg_tensor->get_max_value();
  if (min_value != nullptr && max_value != nullptr) {
    (*dic)[ATTR_MIN_VALUE] = BuildValue(min_value);
    (*dic)[ATTR_MAX_VALUE] = BuildValue(max_value);
  }

  (*dic)[ATTR_DTYPE] = arg_tensor->BuildType();
  (*dic)[ATTR_VALUE] = BuildValue(arg_tensor->BuildValue());
}

void ConvertAbstractFunctionToPython(const AbstractBasePtr &abs_base, py::dict *dic) {
  MS_EXCEPTION_IF_NULL(dic);
  MS_EXCEPTION_IF_NULL(abs_base);
  (*dic)[ATTR_SHAPE] = py::none();
  (*dic)[ATTR_DTYPE] = abs_base->BuildType();
  (*dic)[ATTR_VALUE] = py::none();
  if (abs_base->isa<PartialAbstractClosure>()) {
    AbstractBasePtrList args = abs_base->cast<PartialAbstractClosurePtr>()->args();
    if (!args.empty()) {
      MS_EXCEPTION_IF_NULL(args[0]->BuildValue());
      auto value = args[0]->BuildValue()->cast<parse::ClassTypePtr>();
      if (value != nullptr) {
        (*dic)[ATTR_DTYPE] = std::make_shared<TypeType>();
        (*dic)[ATTR_VALUE] = value->obj();
      }
    }
  }
}

bool CheckType(const TypePtr &expected_type, const TypePtr &x) {
  // As x and predicate both are mindspore type statically, here we only to judge whether
  // x is predicate or is a subclass of predicate.
  return IsIdentidityOrSubclass(x, expected_type);
}

// Join all types in args_type_list;
TypePtr TypeJoin(const TypePtrList &args_type_list) {
  if (args_type_list.empty()) {
    MS_LOG(EXCEPTION) << "args_type_list is empty";
  }

  TypePtr type_tmp = args_type_list[0];
  for (std::size_t i = 1; i < args_type_list.size(); i++) {
    type_tmp = abstract::TypeJoin(type_tmp, args_type_list[i]);
  }
  return type_tmp;
}

TypePtr CheckTypeList(const TypePtr &predicate, const TypePtrList &args_type_list) {
  MS_EXCEPTION_IF_NULL(predicate);
  for (const auto &arg_type : args_type_list) {
    MS_EXCEPTION_IF_NULL(arg_type);
    if (!CheckType(predicate, arg_type)) {
      MS_LOG(EXCEPTION) << "The expected is " << predicate->ToString() << ", not " << arg_type->ToString();
    }
  }
  return TypeJoin(args_type_list);
}
}  // namespace

py::dict ConvertAbstractToPython(const AbstractBasePtr &abs_base, bool only_convert_value) {
  MS_EXCEPTION_IF_NULL(abs_base);
  auto dic = py::dict();
  if (abs_base->isa<AbstractTensor>()) {
    ConvertAbstractTensorToPython(abs_base, only_convert_value, &dic);
  } else if (abs_base->isa<AbstractScalar>() || abs_base->isa<AbstractType>() || abs_base->isa<AbstractRefKey>()) {
    ShapeVector shape;
    dic[ATTR_SHAPE] = shape;
    dic[ATTR_DTYPE] = abs_base->BuildType();
    dic[ATTR_VALUE] = BuildValue(abs_base->BuildValue());
  } else if (abs_base->isa<AbstractTuple>()) {
    return AbstractTupleToPython(abs_base, only_convert_value);
  } else if (abs_base->isa<AbstractList>()) {
    return AbstractListToPython(abs_base, only_convert_value);
  } else if (abs_base->isa<AbstractSlice>()) {
    auto arg_slice = dyn_cast<AbstractSlice>(abs_base);
    ShapeVector shape;
    dic[ATTR_SHAPE] = shape;
    dic[ATTR_DTYPE] = arg_slice->BuildType();
    dic[ATTR_VALUE] = BuildValue(arg_slice->BuildValue());
  } else if (abs_base->isa<AbstractRowTensor>()) {
    auto arg = dyn_cast<AbstractRowTensor>(abs_base);
    dic[ATTR_SHAPE] = arg->shape()->shape();
    dic[ATTR_DTYPE] = arg->BuildType();
    dic[ATTR_VALUE] = BuildValue(arg->BuildValue());
  } else if (abs_base->isa<AbstractCOOTensor>()) {
    auto arg = dyn_cast<AbstractCOOTensor>(abs_base);
    dic[ATTR_SHAPE] = arg->shape()->shape();
    dic[ATTR_DTYPE] = arg->BuildType();
    dic[ATTR_VALUE] = BuildValue(arg->BuildValue());
  } else if (abs_base->isa<AbstractCSRTensor>()) {
    auto arg = dyn_cast<AbstractCSRTensor>(abs_base);
    dic[ATTR_SHAPE] = arg->shape()->shape();
    dic[ATTR_DTYPE] = arg->BuildType();
    dic[ATTR_VALUE] = BuildValue(arg->BuildValue());
  } else if (abs_base->isa<AbstractEllipsis>()) {
    dic[ATTR_SHAPE] = py::none();
    dic[ATTR_DTYPE] = py::ellipsis();
    dic[ATTR_VALUE] = py::ellipsis();
  } else if (abs_base->isa<AbstractNone>()) {
    dic[ATTR_SHAPE] = py::none();
    dic[ATTR_DTYPE] = py::none();
    dic[ATTR_VALUE] = py::none();
  } else if (abs_base->isa<AbstractFunction>()) {
    ConvertAbstractFunctionToPython(abs_base, &dic);
  } else if (abs_base->isa<AbstractUndetermined>()) {
    auto arg = dyn_cast<AbstractUndetermined>(abs_base);
    dic[ATTR_SHAPE] = py::none();
    dic[ATTR_DTYPE] = arg->BuildType();
    dic[ATTR_VALUE] = py::none();
  } else if (abs_base->isa<AbstractMonad>()) {
    dic[ATTR_SHAPE] = py::none();
    dic[ATTR_DTYPE] = abs_base->BuildType();
    dic[ATTR_VALUE] = py::none();
  } else {
    auto value = abs_base->BuildValue();
    MS_EXCEPTION_IF_NULL(value);
    if ((*value == *kAnyValue)) {
      auto value_desc = abs_base->value_desc();
      MS_EXCEPTION(TypeError) << "Unsupported parameter " << (value_desc.empty() ? "type" : value_desc)
                              << " for python primitive." << abs_base->ToString();
    }
    MS_EXCEPTION(TypeError) << "Unsupported parameter type for python primitive, the parameter value is "
                            << value->ToString();
  }
  return dic;
}

namespace {
py::tuple PreparePyInputs(const PrimitivePyPtr &, const AbstractBasePtrList &args) {
  // The monad parameter is defined at the end of the parameter and needs to be ignored
  std::size_t size_args = args.size() - GetAbstractMonadNum(args);
  py::tuple py_args(size_args);
  for (size_t i = 0; i < size_args; i++) {
    auto arg_i = (args)[i];
    py_args[i] = ConvertAbstractToPython(arg_i);
  }
  return py_args;
}

void CheckCustomPrimOutputInferResult(const PrimitivePtr &prim, const AbstractBasePtr &res_spec) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(res_spec);
  const string kOutputNum = "output_num";
  if (prim->IsCustomPrim()) {
    // Raise error if output_num is not match the infer result.
    auto output_num_value = prim->GetAttr(kOutputNum);
    if (output_num_value == nullptr) {
      MS_LOG(DEBUG) << "The output num may no need to check";
      return;
    }
    int64_t output_num = GetValue<int64_t>(output_num_value);
    if (res_spec->isa<AbstractTensor>() && output_num != 1) {
      MS_LOG(EXCEPTION) << "Custom operator primitive[" << prim->ToString()
                        << "]'s attribute[output_num]:" << output_num << " not matches the infer result "
                        << res_spec->ToString();
    } else if (res_spec->isa<AbstractTuple>() &&
               (res_spec->cast<AbstractTuplePtr>()->size() != LongToSize(output_num))) {
      MS_LOG(EXCEPTION) << "Custom operator primitive[" << prim->ToString()
                        << "]'s attribute[output_num]:" << output_num << " not matches the infer result "
                        << res_spec->ToString();
    }
  }
}

void SetValueRange(const AbstractBasePtr &tensor, const py::object &output) {
  if (output.is_none()) {
    return;
  }
  py::object obj_min =
    output.contains(py::str(ATTR_MIN_VALUE)) ? (py::object)output[ATTR_MIN_VALUE] : (py::object)py::none();
  py::object obj_max =
    output.contains(py::str(ATTR_MAX_VALUE)) ? (py::object)output[ATTR_MAX_VALUE] : (py::object)py::none();
  if (!obj_min.is_none() && !obj_max.is_none()) {
    bool converted = true;
    ValuePtr min_value = nullptr;
    ValuePtr max_value = nullptr;
    converted = parse::ConvertData(obj_min, &min_value);
    if (!converted) {
      MS_LOG(EXCEPTION) << "Convert shape min value data failed";
    }
    converted = parse::ConvertData(obj_max, &max_value);
    if (!converted) {
      MS_LOG(EXCEPTION) << "Convert shape max value data failed";
    }
    auto abs_tensor = dyn_cast<abstract::AbstractTensor>(tensor);
    abs_tensor->set_value_range(min_value, max_value);
  }
}

static bool IsMonadType(const py::object &type_obj) {
  if (py::isinstance<Type>(type_obj)) {
    auto type = type_obj.cast<Type *>();
    return type->isa<MonadType>();
  }
  return false;
}

AbstractBasePtr ToMonadAbstract(const py::object &type_obj) {
  if (py::isinstance<Type>(type_obj)) {
    auto type = type_obj.cast<Type *>();
    if (!type->isa<MonadType>()) {
      MS_LOG(EXCEPTION) << "Not a monad type object: " << py::str(type_obj);
    }
    return abstract::MakeMonadAbstract(type->cast<MonadTypePtr>());
  }
  MS_LOG(EXCEPTION) << "Not a type object: " << py::str(type_obj);
}

py::object GetPyAbsItemOfTupleOut(const py::object &output, const size_t index) {
  auto out_dict = output.cast<py::dict>();
  auto type_obj = out_dict[ATTR_DTYPE];
  auto shape_obj = out_dict[ATTR_SHAPE];
  auto out_item = py::dict();
  auto shape_tuple = shape_obj.cast<py::tuple>();
  auto typeid_tuple = type_obj.cast<py::tuple>();
  out_item[ATTR_DTYPE] = typeid_tuple[index];
  out_item[ATTR_SHAPE] = shape_tuple[index];
  if (output.contains(py::str(ATTR_MIN_SHAPE))) {
    out_item[ATTR_MIN_SHAPE] = output[ATTR_MIN_SHAPE].cast<py::tuple>()[index];
  }
  if (output.contains(py::str(ATTR_MAX_SHAPE))) {
    out_item[ATTR_MAX_SHAPE] = output[ATTR_MAX_SHAPE].cast<py::tuple>()[index];
  }
  out_item[ATTR_VALUE] = py::none();
  return out_item;
}

AbstractBasePtr MakePyInferRes2AbstractTensor(const py::object &shape_obj, const py::object &type_obj,
                                              const py::object &output) {
  auto ret_vec = shape_obj.cast<ShapeVector>();
  auto ret_dtype = type_obj.cast<TypePtr>();
  ShapeVector min_shape_vec;
  ShapeVector max_shape_vec;

  if (!output.is_none()) {
    py::object min_shape =
      output.contains(py::str(ATTR_MIN_SHAPE)) ? (py::object)output[ATTR_MIN_SHAPE] : (py::object)py::none();
    py::object max_shape =
      output.contains(py::str(ATTR_MAX_SHAPE)) ? (py::object)output[ATTR_MAX_SHAPE] : (py::object)py::none();
    if (!min_shape.is_none()) {
      min_shape_vec = min_shape.cast<ShapeVector>();
    }
    if (!max_shape.is_none()) {
      max_shape_vec = max_shape.cast<ShapeVector>();
    }
  }

  auto ret_shape = std::make_shared<abstract::Shape>(ret_vec, min_shape_vec, max_shape_vec);
  AbstractBasePtr tensor = MakeAbstractTensor(ret_shape, ret_dtype);

  SetValueRange(tensor, output);
  return tensor;
}

AbstractBasePtr MakePyInferRes2Abstract(const py::object &output) {
  auto out_dict = output.cast<py::dict>();
  auto type_obj = out_dict[ATTR_DTYPE];
  auto shape_obj = out_dict[ATTR_SHAPE];
  if ((py::isinstance<py::list>(shape_obj) || py::isinstance<py::tuple>(shape_obj)) && py::isinstance<Type>(type_obj)) {
    auto ret_vec = shape_obj.cast<ShapeVector>();
    auto ret_dtype = type_obj.cast<TypePtr>();
    MS_EXCEPTION_IF_NULL(ret_dtype);
    // if the size of shape list is empty, return an scalar abstract
    if (ret_vec.empty() && (!ret_dtype->isa<TensorType>())) {
      abstract::AbstractScalarPtr abs_scalar = std::make_shared<abstract::AbstractScalar>(kAnyValue, ret_dtype);
      return abs_scalar;
    }
    return MakePyInferRes2AbstractTensor(shape_obj, type_obj, output);
  } else if (py::isinstance<py::tuple>(shape_obj) && py::isinstance<py::tuple>(type_obj)) {
    auto typeid_tuple = type_obj.cast<py::tuple>();
    AbstractBasePtrList ptr_list;
    for (size_t it = 0; it < typeid_tuple.size(); ++it) {
      auto output_it = GetPyAbsItemOfTupleOut(output, it);
      auto tensor_it = MakePyInferRes2Abstract(output_it);
      ptr_list.push_back(tensor_it);
    }
    auto tuple = std::make_shared<abstract::AbstractTuple>(ptr_list);
    return tuple;
  } else if (py::isinstance<py::list>(shape_obj) && py::isinstance<py::list>(type_obj)) {
    auto typeid_list = type_obj.cast<py::list>();
    AbstractBasePtrList ptr_list;
    for (size_t it = 0; it < typeid_list.size(); ++it) {
      auto output_it = GetPyAbsItemOfTupleOut(output, it);
      auto tensor_it = MakePyInferRes2Abstract(output_it);
      ptr_list.push_back(tensor_it);
    }
    auto list = std::make_shared<abstract::AbstractList>(ptr_list);
    return list;
  } else if (shape_obj.is_none() && type_obj.is_none()) {
    // AbstractNone indicates there is no output for this CNode node.
    auto abstract_none = std::make_shared<abstract::AbstractNone>();
    return abstract_none;
  } else if (IsMonadType(type_obj)) {
    // Return monad abstract if it is monad type.
    return ToMonadAbstract(type_obj);
  } else {
    // When sparse enabled, the undetermined might be raised and eliminated in opt passes
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    bool enable_sparse = context->get_param<bool>(MS_CTX_ENABLE_SPARSE);
    if (enable_sparse) {
      return std::make_shared<abstract::AbstractUndetermined>();
    }
    MS_LOG(EXCEPTION) << "Python evaluator return invalid shape or type. " << (std::string)py::str(type_obj);
  }
}

AbstractBasePtr PyInferRes2Abstract(const PrimitivePyPtr &prim_py, const py::dict &output) {
  // Convert to AbstractValue based on type and shape
  if (output[ATTR_VALUE].is_none()) {
    return MakePyInferRes2Abstract(output);
  }

  // Convert pyobject to Value, then to AbstractValue
  auto out_dtype = output[ATTR_DTYPE];
  TypePtr dtype = py::isinstance<Type>(out_dtype) ? out_dtype.cast<TypePtr>() : nullptr;
  ValuePtr converted_ret = nullptr;
  bool converted = parse::ConvertData(output[ATTR_VALUE], &converted_ret, false, dtype);
  if (!converted) {
    MS_LOG(EXCEPTION) << "Convert data failed";
  }
  auto res_spec = FromValue(converted_ret);
  MS_EXCEPTION_IF_NULL(res_spec);
  if (res_spec->isa<AbstractTensor>()) {
    // Replace to tensor constant node in specialize
    auto res_tensor = res_spec->cast<AbstractTensorPtr>();
    res_tensor->set_value(converted_ret);
    SetValueRange(res_tensor, output);
  }
  CheckCustomPrimOutputInferResult(prim_py, res_spec);
  return res_spec;
}
}  // namespace

EvalResultPtr StandardPrimEvaluator::RunPyInferValue(const AnalysisEnginePtr &engine, const AbstractBasePtr &abs_base,
                                                     const AbstractBasePtrList &args) {
  auto prim_py = dyn_cast<PrimitivePy>(prim_);
  if (prim_py == nullptr) {
    MS_LOG(EXCEPTION) << "The primitive with type 'kPrimTypePyCheck' should be a python primitive.";
  }
  // Call checking method 'infer_value' for python primitive
  MS_LOG(DEBUG) << "Begin input args checking for: " << prim_py->ToString();
  auto py_args = PreparePyInputs(prim_py, args);
  py::tuple py_vals(py_args.size());
  auto added_attrs = prim_->evaluate_added_attrs();
  for (size_t i = 0; i < py_args.size(); ++i) {
    py_vals[i] = py_args[i][ATTR_VALUE];
  }
  py::object py_ret = prim_py->RunInferValue(py_vals);
  if (py::isinstance<py::none>(py_ret)) {
    return std::make_shared<EvalResult>(abs_base, std::make_shared<AttrValueMap>(added_attrs));
  }
  // Convert pyobject to Value, then to AbstractValue
  ValuePtr converted_ret = nullptr;
  TypePtr dtype = abs_base->BuildType();
  bool converted = parse::ConvertData(py_ret, &converted_ret, false, dtype);
  if (!converted) {
    MS_LOG(EXCEPTION) << "Convert data failed";
  }
  auto res_spec = FromValue(converted_ret);
  MS_EXCEPTION_IF_NULL(res_spec);
  if (res_spec->isa<AbstractTensor>()) {
    // Replace to tensor constant node in specialize
    auto res_tensor = res_spec->cast<AbstractTensorPtr>();
    res_tensor->set_value(converted_ret);
  }
  return std::make_shared<EvalResult>(res_spec, std::make_shared<AttrValueMap>(added_attrs));
}

// Apply EvalResult from cached result for a given primitive.
static inline EvalResultPtr ApplyCacheEvalResult(const PrimitivePtr &prim, const EvalResultPtr &result) {
  auto &attrs = result->attribute();
  if (attrs != nullptr) {
    prim->set_evaluate_added_attrs(*attrs);
  }
  return std::make_shared<EvalResult>(result->abstract()->Clone(), attrs);
}

EvalResultPtr StandardPrimEvaluator::EvalPyCheckPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args) {
  // Try to get infer result from evaluator cache.
  auto eval_result = evaluator_cache_mgr_->GetValue(args);
  if (eval_result != nullptr) {
    // Evaluator cache hit.
    return std::make_shared<EvalResult>(eval_result->abstract()->Clone(), eval_result->attribute());
  }
  // In pynative mode (engine == nullptr), it is difficult to set added_attrs to
  // python object by C++ code, so we disable global eval cache in pynative mode.
  const bool enable_global_cache = (engine != nullptr);
  if (enable_global_cache) {
    // Try to get infer result from global primitive evaluate cache.
    eval_result = eval_cache_->Get(prim_, args);
    if (eval_result != nullptr) {
      // Global primitive evaluate cache hit.
      evaluator_cache_mgr_->SetValue(args, eval_result);
      return ApplyCacheEvalResult(prim_, eval_result);
    }
  }
  // PrimitivePy is expected for EvalPyCheckPrim.
  auto prim_py = dyn_cast<PrimitivePy>(prim_);
  if (prim_py == nullptr) {
    MS_LOG(EXCEPTION) << "The primitive with type 'kPrimTypePyCheck' should be a python primitive.";
  }
  // We should copy attributes before running check and infer,
  // since they may be changed during check and infer.
  auto input_attrs = prim_py->attrs();
  prim_py->BeginRecordAddAttr();
  auto py_args = PreparePyInputs(prim_py, args);
  // Call checking method '__check__' for subclass of 'PrimitiveWithCheck'.
  prim_py->RunCheck(py_args);
  auto abs = eval_impl_.infer_shape_impl_(engine, prim_py, args);
  prim_py->EndRecordAddAttr();
  auto &added_attrs = prim_py->evaluate_added_attrs();
  eval_result = std::make_shared<EvalResult>(abs, std::make_shared<AttrValueMap>(added_attrs));
  if (py::hasattr(prim_py->GetPyObj(), PY_PRIM_METHOD_INFER_VALUE)) {
    // Call 'infer_value()' method if it is exsited, for constant propagation.
    eval_result = RunPyInferValue(engine, eval_result->abstract(), args);
  }
  // Save infer result to caches (evaluator cache and global cache).
  if (enable_global_cache) {
    eval_cache_->Put(prim_py, std::move(input_attrs), args, eval_result);
  }
  evaluator_cache_mgr_->SetValue(args, eval_result);
  return eval_result;
}

namespace {
void CheckSequenceArgumentForCppPrimitive(const PrimitivePtr &prim, const AbstractBasePtrList &args) {
  // To check tuple/list operations with a white list of Python primitive.
  auto iter = prims_transparent_pass_sequence.find(prim->name());
  if (iter == prims_transparent_pass_sequence.end()) {
    // The primitive use all elements of each argument.
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i]->isa<abstract::AbstractSequence>()) {
        MS_LOG(DEBUG) << "Primitive \'" << prim->name() << "\' is consuming tuple/list arguments[" << i
                      << "]: " << args[i]->ToString();
        SetSequenceElementsUseFlags(args[i], true);
      }
    }
    return;
  }

  // It's transparent pass primitive or using partial elements primitive.
  auto index_list = iter->second;
  if (index_list.empty()) {
    MS_LOG(EXCEPTION) << "The primitive list should not be empty for " << prim->name();
  }
  // Ignore all arguments, no need checking if AbstractSequence.
  if (index_list[0] == -1) {
    return;
  }
  // Check the specific arguments index.
  for (size_t i = 0; i < args.size(); ++i) {
    if (!args[i]->isa<abstract::AbstractSequence>()) {
      continue;
    }
    if (std::find(index_list.begin(), index_list.end(), i) == index_list.end()) {
      // For current tuple/list argument, it's not a primitive of total transparent pass or partial element use.
      MS_LOG(DEBUG) << "Primitive \'" << prim->name() << "\' is consuming specific tuple/list arguments[" << i
                    << "]: " << args[i]->ToString();
      SetSequenceElementsUseFlags(args[i], true);
    }
  }
}

void CheckSequenceArgumentForPythonPrimitive(const PrimitivePtr &prim, const AbstractBasePtrList &args) {
  // Consider all primitive implemented python infer() real use the tuple/list arguments.
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i]->isa<abstract::AbstractSequence>()) {
      MS_LOG(DEBUG) << "Primitive \'" << prim->name() << "\' is consuming tuple/list arguments[" << i
                    << "]: " << args[i]->ToString();
      SetSequenceElementsUseFlags(args[i], true);
    }
  }
}
}  // namespace

EvalResultPtr StandardPrimEvaluator::EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args) {
  // To check tuple/list operations with a white list of Python primitive.
  CheckSequenceArgumentForCppPrimitive(prim_, args);

  if (prims_to_skip_undetermined_infer.find(prim_->name()) == prims_to_skip_undetermined_infer.end()) {
    auto ret_abstract = AbstractEval(args);
    if (ret_abstract != nullptr) {
      MS_LOG(DEBUG) << "StandardPrimEvaluator eval Undetermined";
      return ret_abstract;
    }
  }
  if (prim_->prim_type() == PrimType::kPrimTypePyCheck) {
    return EvalPyCheckPrim(engine, args);
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool need_infer_value = !eval_impl_.in_white_list_;
  if (need_infer_value == false) {
    need_infer_value = ((context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode)) &&
                       std::all_of(args.begin(), args.end(), [](const AbstractBasePtr &abs) -> bool {
                         MS_EXCEPTION_IF_NULL(abs);
                         auto value = abs->BuildValue();
                         return (value != nullptr && !value->isa<AnyValue>() && !value->isa<None>() &&
                                 !value->isa<Monad>() && !value->isa<FuncGraph>());
                       });
  }
  AbstractBasePtr abs_base = nullptr;
  ValuePtr value = nullptr;
  prim_->BeginRecordAddAttr();
  if (need_infer_value && eval_impl_.infer_value_impl_ != nullptr) {
    value = eval_impl_.infer_value_impl_(prim_, args);
    if (value != nullptr) {
      abs_base = value->ToAbstract();
      prim_->EndRecordAddAttr();
      auto added_attrs = prim_->evaluate_added_attrs();
      return std::make_shared<EvalResult>(abs_base, std::make_shared<AttrValueMap>(added_attrs));
    }
  }
  abs_base = eval_impl_.infer_shape_impl_(engine, prim_, args);
  prim_->EndRecordAddAttr();
  const auto &added_attrs = prim_->evaluate_added_attrs();
  return std::make_shared<EvalResult>(abs_base, std::make_shared<AttrValueMap>(added_attrs));
}

EvalResultPtr PythonPrimEvaluator::EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args) {
  // Consider all primitive implemented python infer() real use the tuple/list arguments.
  CheckSequenceArgumentForPythonPrimitive(prim_py_, args);

  // Ensure input arguments are evaluated.
  auto ret_abstract = AbstractEval(args);
  if (ret_abstract != nullptr) {
    MS_LOG(DEBUG) << "PythonPrimEvaluator eval Undetermined";
    return ret_abstract;
  }
  // Try to get infer result from evaluator cache.
  auto eval_result = evaluator_cache_mgr_->GetValue(args);
  if (eval_result != nullptr) {
    return std::make_shared<EvalResult>(eval_result->abstract()->Clone(), eval_result->attribute());
  }
  // In pynative mode (engine == nullptr), it is difficult to set added_attrs to
  // python object by C++ code, so we disable global eval cache in pynative mode.
  const bool enable_global_cache = (engine != nullptr);
  if (enable_global_cache) {
    // Try to get infer result from global primitive eval cache.
    eval_result = eval_cache_->Get(prim_py_, args);
    if (eval_result != nullptr) {
      // Global cache hit.
      evaluator_cache_mgr_->SetValue(args, eval_result);
      return ApplyCacheEvalResult(prim_py_, eval_result);
    }
  }
  // Cache miss, run infer. We should copy attributes before
  // running infer, since they may be changed during infer.
  auto input_attrs = prim_py_->attrs();
  auto py_args = PreparePyInputs(prim_py_, args);
  prim_py_->BeginRecordAddAttr();
  py::dict output = prim_py_->RunInfer(py_args);
  prim_py_->EndRecordAddAttr();
  const auto &added_attrs = prim_py_->evaluate_added_attrs();
  MS_LOG(DEBUG) << "Output type is " << (std::string)py::str(output);
  auto res_abs = PyInferRes2Abstract(prim_py_, output);
  MS_LOG(DEBUG) << "Python InferTensor result abstract: " << res_abs->ToString();
  eval_result = std::make_shared<EvalResult>(res_abs, std::make_shared<AttrValueMap>(added_attrs));
  // Save result to global primitive eval cache.
  if (enable_global_cache) {
    eval_cache_->Put(prim_py_, std::move(input_attrs), args, eval_result);
  }
  evaluator_cache_mgr_->SetValue(args, eval_result);
  return eval_result;
}

EvalResultPtr UniformPrimEvaluator::EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args) {
  auto ret_abstract = AbstractEval(args);
  if (ret_abstract != nullptr) {
    MS_LOG(DEBUG) << "UniformPrimEvaluator eval Undetermined";
    return ret_abstract;
  }
  // if func_desc_.retval type is super class of parameter type, then make the retval type as parameter type.
  if (nargs_ != args.size()) {
    MS_LOG(EXCEPTION) << "UniformPrimEvaluator expect " << nargs_ << " args, but got " << args.size() << " inputs";
  }
  TypePtr ret_value_type = return_value_type_;
  ValuePtrList value_list;
  for (const auto &arg : args) {
    // Check if all arguments are scalar type.
    MS_EXCEPTION_IF_NULL(arg);
    if (arg->isa<AbstractScalar>()) {
      auto arg_scalar = dyn_cast<AbstractScalar>(arg);
      auto arg_value = arg_scalar->GetValueTrack();
      value_list.push_back(arg_value);
    } else {
      // Raise TypeError Expected Scalar.
      MS_LOG(EXCEPTION) << "Expect scalar arguments for uniform primitives.";
    }
  }
  for (const auto &item : type_map_) {
    TypePtrList selections;
    MS_EXCEPTION_IF_NULL(item.second);
    (void)std::transform(item.second->begin(), item.second->end(), std::back_inserter(selections),
                         [&args](size_t arg_idx) -> TypePtr {
                           if (arg_idx >= args.size()) {
                             MS_LOG(EXCEPTION) << "Index:" << arg_idx << " out of range:" << args.size();
                           }
                           MS_EXCEPTION_IF_NULL(args[arg_idx]);
                           return args[arg_idx]->GetTypeTrack();
                         });
    TypePtr res = CheckTypeList(item.first, selections);
    MS_EXCEPTION_IF_NULL(return_value_type_);
    MS_EXCEPTION_IF_NULL(item.first);
    if (*return_value_type_ == *(item.first)) {
      ret_value_type = res;
    }
  }

  ValuePtr evaluated_value = RunImpl(value_list);
  if (!(*evaluated_value == *kAnyValue)) {
    ret_value_type = evaluated_value->type();
  }
  // for comparison primitives , return type shall have be specified to be bool.
  if (specify_out_type_ != nullptr) {
    ret_value_type = specify_out_type_;
  }

  AbstractScalarPtr abs_base = std::make_shared<AbstractScalar>(evaluated_value, ret_value_type);
  return std::make_shared<EvalResult>(abs_base, std::make_shared<AttrValueMap>());
}

ValuePtr UniformPrimEvaluator::RunImpl(const ValuePtrList &args) const {
  if (!eval_value_) {
    return kAnyValue;
  } else {
    if (std::any_of(args.begin(), args.end(), [](const ValuePtr &arg) {
          MS_EXCEPTION_IF_NULL(arg);
          return arg->isa<AnyValue>();
        })) {
      return kAnyValue;
    }
    return impl_(args);
  }
}

// Primitive implementation
// static function start
namespace {
EvaluatorPtr InitStandardPrimEvaluator(PrimitivePtr primitive, const StandardPrimitiveImplReg eval_impl) {
  EvaluatorPtr prim_evaluator = std::make_shared<StandardPrimEvaluator>(primitive, eval_impl);
  return prim_evaluator;
}

EvaluatorPtr InitUniformPrimEvaluator(const PrimitivePtr &primitive, PrimitiveImpl prim_impl, bool eval_value,
                                      const TypePtr &specify_out_type) {
  FunctionPtr func = nullptr;
  (void)prim::PrimToFunction::GetInstance().GetFunction(primitive, &func);
  MS_EXCEPTION_IF_NULL(func);

  EvaluatorPtr uniform_primitive_evaluator =
    std::make_shared<UniformPrimEvaluator>(func, prim_impl, eval_value, specify_out_type);
  return uniform_primitive_evaluator;
}

FuncGraphPtr PyObjToGraph(const AnalysisEnginePtr &engine, const ValuePtr &method) {
  MS_EXCEPTION_IF_NULL(engine);
  MS_EXCEPTION_IF_NULL(method);
  if (!method->isa<parse::PyObjectWrapper>()) {
    MS_LOG(EXCEPTION) << "Method type error: " << method->ToString();
  }

  std::shared_ptr<PyObjectWrapper> obj = method->cast<std::shared_ptr<PyObjectWrapper>>();
  FuncGraphPtr func_graph = mindspore::parse::ConvertToFuncGraph(obj->obj());
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Parse python object: " << method->ToString() << " failed";
  }

  FuncGraphManagerPtr manager = engine->func_graph_manager();
  manager->AddFuncGraph(func_graph);
  return func_graph;
}

inline void AddToManager(const AnalysisEnginePtr &engine, const FuncGraphPtr func_graph) {
  MS_EXCEPTION_IF_NULL(engine);
  FuncGraphManagerPtr manager = engine->func_graph_manager();
  manager->AddFuncGraph(func_graph);
}

enum class REQUIRE_TYPE { ATTR, METHOD };

EvalResultPtr StaticGetterInferred(const ValuePtr &value, const ConfigPtr &data_conf, const AnfNodeConfigPtr &old_conf,
                                   REQUIRE_TYPE require_type = REQUIRE_TYPE::METHOD) {
  MS_EXCEPTION_IF_NULL(old_conf);
  AbstractBasePtr abstract = ToAbstract(value, AnalysisContext::DummyContext(), old_conf);
  AbstractFunctionPtr abs_func = dyn_cast<abstract::AbstractFunction>(abstract);
  MS_EXCEPTION_IF_NULL(abs_func);

  // Create new cnode
  std::vector<AnfNodePtr> input = {NewValueNode(prim::kPrimPartial)};
  auto func_graph_func = dyn_cast<abstract::FuncGraphAbstractClosure>(abs_func);
  if (func_graph_func != nullptr) {
    FuncGraphPtr fg = func_graph_func->func_graph();
    input.push_back(NewValueNode(fg));
  } else {
    auto prim_func = dyn_cast<abstract::PrimitiveAbstractClosure>(abs_func);
    MS_EXCEPTION_IF_NULL(prim_func);
    PrimitivePtr prim = prim_func->prim();
    input.push_back(NewValueNode(prim));
  }

  AnfNodeConfigPtr conf = dyn_cast<abstract::AnfNodeConfig>(data_conf);
  MS_EXCEPTION_IF_NULL(conf);
  input.push_back(conf->node());
  MS_EXCEPTION_IF_NULL(old_conf);
  FuncGraphPtr func_graph = old_conf->node()->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  CNodePtr new_cnode = func_graph->NewCNode(input);
  if (require_type == REQUIRE_TYPE::ATTR) {
    new_cnode = func_graph->NewCNode({new_cnode});
  }
  AnalysisEnginePtr eng = old_conf->engine();
  AnfNodeConfigPtr fn_conf = eng->MakeConfig(new_cnode, old_conf->context(), old_conf->func_graph());
  return eng->ForwardConfig(old_conf, fn_conf);
}

EvalResultPtr GetEvaluatedValueForNameSpaceString(const AnalysisEnginePtr &, const AbstractBasePtrList &args_spec_list,
                                                  const AnfNodeConfigPtr &out_conf) {
  // args_spec_list: same as StaticGetter
  if (args_spec_list.size() < 2) {
    MS_LOG(EXCEPTION) << "Size of args_spec_list is less than 2";
  }
  MS_EXCEPTION_IF_NULL(out_conf);
  // An external type.
  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  MS_EXCEPTION_IF_NULL(args_spec_list[1]);
  auto data_value = args_spec_list[0]->BuildValue();
  MS_EXCEPTION_IF_NULL(data_value);
  if (!data_value->isa<parse::NameSpace>()) {
    MS_EXCEPTION(TypeError) << "Not supported to get attribute for " << data_value->ToString()
                            << "\nThe first argument should be a NameSpace, but got " << args_spec_list[0]->ToString();
  }

  auto item_value = args_spec_list[1]->BuildValue();
  MS_EXCEPTION_IF_NULL(item_value);
  if (item_value->isa<StringImm>()) {
    item_value = std::make_shared<parse::Symbol>(item_value->cast<StringImmPtr>()->value());
  }

  if (!item_value->isa<parse::Symbol>()) {
    MS_LOG(EXCEPTION) << "The value of the attribute could not be inferred: " << item_value->ToString();
  }

  // item_name to func addr from obj_map
  parse::SymbolPtr symbol = item_value->cast<parse::SymbolPtr>();
  parse::NameSpacePtr name_space = data_value->cast<parse::NameSpacePtr>();
  MS_EXCEPTION_IF_NULL(out_conf);
  auto out_node = out_conf->node();
  FuncGraphPtr func_graph = out_node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto new_node = parse::ResolveSymbol(func_graph->manager(), name_space, symbol, out_node);
  if (new_node == nullptr) {
    MS_LOG(EXCEPTION) << "Resolve node failed";
  }
  if (pipeline::GetJitLevel() == "o0" && IsValueNode<FuncGraph>(new_node)) {
    UpdateDebugInfo(GetValueNode<FuncGraphPtr>(new_node), out_node->scope(), out_node->debug_info());
  }

  // Replace old node with the resolved new node in order list.
  func_graph->ReplaceInOrder(out_node, new_node);

  AnalysisEnginePtr eng = out_conf->engine();
  MS_EXCEPTION_IF_NULL(eng);
  AnfNodeConfigPtr fn_conf = eng->MakeConfig(new_node, out_conf->context(), out_conf->func_graph());
  return eng->ForwardConfig(out_conf, fn_conf);
}

EvalResultPtr GetEvaluatedValueForClassAttrOrMethod(const AnalysisEnginePtr &engine,
                                                    const AbstractBasePtrList &args_spec_list,
                                                    const ValuePtr &item_value, const ConfigPtr &data_conf,
                                                    const AnfNodeConfigPtr &out_conf) {
  if (args_spec_list.empty()) {
    MS_LOG(EXCEPTION) << "args_spec_list is empty";
  }
  AbstractClassPtr cls = CheckArg<AbstractClass>("__FUNC__", args_spec_list, 0);

  // If item_value is an attribute, get abstract value from AbstractClass
  MS_EXCEPTION_IF_NULL(item_value);
  if (!item_value->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "Attribute type error";
  }
  std::string item_name = item_value->cast<StringImmPtr>()->value();
  MS_LOG(DEBUG) << "Resolve name: " << cls->tag().name();
  MS_LOG(DEBUG) << "Resolve item: " << item_name;
  AbstractBasePtr attr = cls->GetAttribute(item_name);
  if (attr != nullptr) {
    return std::make_shared<EvalResult>(attr, nullptr);
  }

  ValuePtr method = cls->GetMethod(item_name);
  if (method->isa<AnyValue>()) {
    MS_EXCEPTION_IF_NULL(args_spec_list[0]);
    MS_EXCEPTION_IF_NULL(args_spec_list[0]->BuildType());
    MS_EXCEPTION(AttributeError) << "Unknown field, data type: " << args_spec_list[0]->BuildType()->ToString()
                                 << ", item value: " << item_value->ToString();
  }

  // Infer class method
  ValuePtr converted_value = PyObjToGraph(engine, method);
  return StaticGetterInferred(converted_value, data_conf, out_conf);
}

EvalResultPtr GetEvaluatedValueForBuiltinTypeAttrOrMethod(const AnalysisEnginePtr &engine, const ValuePtr &item_value,
                                                          const TypePtr &data_type, const ConfigPtr &data_conf,
                                                          const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(item_value);
  MS_EXCEPTION_IF_NULL(data_type);
  // The method maybe a Primitive or Composite
  if (!item_value->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "Expect a string, but got: " << item_value->ToString();
  }

  std::string item_name = item_value->cast<StringImmPtr>()->value();
  REQUIRE_TYPE require_type = REQUIRE_TYPE::METHOD;
  Any require = pipeline::Resource::GetMethodPtr(data_type->type_id(), item_name);
  if (require.empty()) {
    require = pipeline::Resource::GetAttrPtr(data_type->type_id(), item_name);
    if (require.empty()) {
      MS_LOG(EXCEPTION) << "MindSpore not support to get attribute \'" << item_name << "\' of a type["
                        << data_type->ToString() << "]";
    }
    require_type = REQUIRE_TYPE::ATTR;
  }

  ValuePtr converted_value = nullptr;
  if (require.is<std::string>()) {
    // composite registered in standard_method_map go to this branch
    converted_value = prim::GetPythonOps(require.cast<std::string>());
    MS_EXCEPTION_IF_NULL(converted_value);
    if (pipeline::GetJitLevel() == "o0" && converted_value->isa<FuncGraph>()) {
      UpdateDebugInfo(converted_value->cast<FuncGraphPtr>(), out_conf->node()->scope(), out_conf->node()->debug_info());
    }
    if (!converted_value->isa<Primitive>()) {
      AddToManager(engine, converted_value->cast<FuncGraphPtr>());
    }
  } else if (require.is<PrimitivePtr>()) {
    converted_value = require.cast<PrimitivePtr>();
  } else {
    MS_LOG(EXCEPTION) << "Expect to get string or PrimitivePtr from attr or method map, but got " << require.ToString();
  }
  return StaticGetterInferred(converted_value, data_conf, out_conf, require_type);
}

enum ResolveType : int64_t {
  kResolveTypeUserDefineClass = 1,
  kResolveTypeBuiltInType,
  kResolveTypeFunction,
};

int64_t GetResolveType(const TypePtr &data_type) {
  MS_EXCEPTION_IF_NULL(data_type);
  if (data_type->type_id() == kObjectTypeClass) {
    return kResolveTypeUserDefineClass;
  }
  // Try to search method map, if not found, the data_type should be External type.
  if (pipeline::Resource::IsTypeInBuiltInMap(data_type->type_id())) {
    return kResolveTypeBuiltInType;
  }
  return kResolveTypeFunction;
}

EvalResultPtr StaticGetter(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list,
                           const ConfigPtr &data_conf, const AnfNodeConfigPtr &out_conf) {
  // Inputs: namespace and its static function; or class and its member function
  CheckArgsSize("StaticGetter", args_spec_list, 2);

  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  MS_EXCEPTION_IF_NULL(args_spec_list[1]);
  MS_LOG(DEBUG) << "Args[0]: " << args_spec_list[0]->ToString();
  MS_LOG(DEBUG) << "Args[1]: " << args_spec_list[1]->ToString();
  TypePtr data_type = args_spec_list[0]->BuildType();
  ValuePtr item_value = args_spec_list[1]->BuildValue();
  ScopePtr scope = kDefaultScope;
  if (out_conf != nullptr) {
    scope = out_conf->node()->scope();
  }
  ScopeGuard scope_guard(scope);
  MS_EXCEPTION_IF_NULL(item_value);
  if (item_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "The value of the attribute could not be inferred: " << item_value->ToString();
  }

  int64_t resolve_type = GetResolveType(data_type);
  if (resolve_type == kResolveTypeUserDefineClass) {
    return GetEvaluatedValueForClassAttrOrMethod(engine, args_spec_list, item_value, data_conf, out_conf);
  } else if (resolve_type == kResolveTypeBuiltInType) {
    return GetEvaluatedValueForBuiltinTypeAttrOrMethod(engine, item_value, data_type, data_conf, out_conf);
  } else {
    return GetEvaluatedValueForNameSpaceString(engine, args_spec_list, out_conf);
  }
}
}  // namespace

namespace {
class EmbedEvaluator : public SymbolicPrimEvaluator {
 public:
  EmbedEvaluator() : SymbolicPrimEvaluator("EmbedEvaluator") {}
  ~EmbedEvaluator() override = default;
  MS_DECLARE_PARENT(EmbedEvaluator, SymbolicPrimEvaluator);
  EvalResultPtr EvalPrim(const ConfigPtrList &args_conf_list) override {
    // arg: free variable to be embedded
    if (args_conf_list.size() != 1) {
      MS_LOG(EXCEPTION) << "EmbedEvaluator requires 1 parameter, but got " << args_conf_list.size();
    }
    AnfNodeConfigPtr node_conf = dyn_cast<AnfNodeConfig>(args_conf_list[0]);
    MS_EXCEPTION_IF_NULL(node_conf);
    MS_EXCEPTION_IF_NULL(node_conf->ObtainEvalResult());
    AbstractBasePtr x = node_conf->ObtainEvalResult()->abstract();
    x = SensitivityTransform(x);
    SymbolicKeyInstancePtr key = std::make_shared<SymbolicKeyInstance>(node_conf->node(), x);
    AbstractScalarPtr abs_scalar = std::make_shared<AbstractScalar>(key, std::make_shared<SymbolicKeyType>());
    return std::make_shared<EvalResult>(abs_scalar, std::make_shared<AttrValueMap>());
  }
};

static AnfNodePtr FindParameterNodeByString(const FuncGraphManagerPtr &manager, const std::string &name) {
  MS_EXCEPTION_IF_NULL(manager);
  auto root_g_set = manager->roots();
  if (root_g_set.size() != 1) {
    return nullptr;
  }
  const FuncGraphPtr &root_g = root_g_set.back();
  for (auto &param_node : root_g->parameters()) {
    auto param = param_node->cast<ParameterPtr>();
    if (param && name == param->name()) {
      return param;
    }
  }
  return nullptr;
}

class RefToEmbedEvaluator : public SymbolicPrimEvaluator {
 public:
  RefToEmbedEvaluator() : SymbolicPrimEvaluator("RefToEmbedEvaluator") {}
  ~RefToEmbedEvaluator() override = default;
  MS_DECLARE_PARENT(RefToEmbedEvaluator, SymbolicPrimEvaluator);
  EvalResultPtr EvalPrim(const ConfigPtrList &args_conf_list) override {
    if (args_conf_list.size() != 1) {
      MS_LOG(ERROR) << "Requires 1 parameter, but has: " << args_conf_list.size();
      return nullptr;
    }
    static TypePtr type = std::make_shared<SymbolicKeyType>();
    auto node_conf = dyn_cast<AnfNodeConfig>(args_conf_list[0]);
    if (node_conf == nullptr) {
      MS_LOG(ERROR) << "Conf should be AnfNodeConfig";
      return nullptr;
    }
    MS_EXCEPTION_IF_NULL(node_conf->ObtainEvalResult());
    AbstractBasePtr abs = node_conf->ObtainEvalResult()->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    AbstractRefPtr ref_abs = abs->cast<AbstractRefPtr>();
    if (ref_abs == nullptr) {
      MS_LOG(ERROR) << "The first parameter of RefToEmbed should be Ref, but " << abs->ToString();
      return nullptr;
    }
    auto key_abs = ref_abs->ref_key();
    if (key_abs == nullptr) {
      MS_LOG(ERROR) << "RefToEmbed input Ref key is nullptr.";
      return nullptr;
    }
    auto key_value = key_abs->BuildValue();
    if (key_value == nullptr) {
      MS_LOG(ERROR) << "RefToEmbed input Ref key value is nullptr.";
      return nullptr;
    }
    // Check if the input of RefEmbed is a weight parameter, if not, don't create the
    // specific SymbolicKey.
    // Notes: when different weight parameter have same type and shape passed as parameter to same funcgraph
    // which has RefToEmbed CNode, that funcgraph will not be specialized to different funcgraph, so the
    // RefToEmbed CNode in that funcgraph also should not be evaluated to specific SymbolicKey.
    // Only after that funcgrpah is inlined, the RefToEmbed CNode should be evaluated to specific SymbolicKey.
    bool ifEmbedIsWeight = false;
    if (node_conf->node() != nullptr && node_conf->node()->isa<Parameter>()) {
      auto param = node_conf->node()->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(param);
      ifEmbedIsWeight = param->has_default();
    }
    auto refkey = key_value->cast<RefKeyPtr>();
    if (refkey == nullptr || !ifEmbedIsWeight) {
      auto ret = std::make_shared<AbstractScalar>(type);
      auto ref_value = ref_abs->ref();
      MS_EXCEPTION_IF_NULL(ref_value);
      return std::make_shared<EvalResult>(ret, std::make_shared<AttrValueMap>());
    }

    std::string name = refkey->tag();
    MS_EXCEPTION_IF_NULL(node_conf->node());
    if (node_conf->node()->func_graph() == nullptr) {
      MS_LOG(EXCEPTION) << "Should not evaluate a ValueNode, node: " << node_conf->node()->DebugString();
    }
    const auto &manager = node_conf->node()->func_graph()->manager();
    auto node = FindParameterNodeByString(manager, name);
    if (node == nullptr) {
      MS_LOG(ERROR) << "RefToEmbed input can't find parameter \"" << name << "\" in graph.";
      return nullptr;
    }
    AbstractBasePtr x = ref_abs->ref();
    x = SensitivityTransform(x);
    std::shared_ptr<SymbolicKeyInstance> key = std::make_shared<SymbolicKeyInstance>(node, x);
    std::shared_ptr<AbstractScalar> abs_scalar = std::make_shared<AbstractScalar>(key, type);
    return std::make_shared<EvalResult>(abs_scalar, std::make_shared<AttrValueMap>());
  }
};

class GetAttrEvaluator : public TransitionPrimEvaluator {
 public:
  GetAttrEvaluator() : TransitionPrimEvaluator("GetAttrEvaluator") {}
  ~GetAttrEvaluator() override = default;
  MS_DECLARE_PARENT(GetAttrEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list,
                         const ConfigPtr &in_conf0, const AnfNodeConfigPtr &out_conf) override {
    constexpr auto kGetAttrArgSize = 2;
    auto ret_abstract = AbstractEval(args_spec_list);
    if (ret_abstract != nullptr) {
      MS_LOG(DEBUG) << "GetAttrEvaluator eval Undetermined";
      return ret_abstract;
    }
    // Inputs: data, item
    if (args_spec_list.size() != kGetAttrArgSize) {
      MS_LOG(EXCEPTION) << "Expected args_spec_list size = 2, but has size:" << args_spec_list.size();
    }
    EvalResultPtr ret = nullptr;
    if (bound_node() != nullptr) {
      TraceGuard trace_guard(std::make_shared<TraceResolve>(bound_node()->debug_info()));
      ret = StaticGetter(engine, args_spec_list, in_conf0, out_conf);
    } else {
      ret = StaticGetter(engine, args_spec_list, in_conf0, out_conf);
    }
    // don't lookup from cache, as different out_conf with same node but different context
    // may add different entry to anfnode_config_map, like getattr primitive;
    evaluator_cache_mgr_->SetValue(args_spec_list, ret);
    return ret;
  }
};

class ResolveEvaluator : public TransitionPrimEvaluator {
 public:
  ResolveEvaluator() : TransitionPrimEvaluator("ResolveEvaluator") {}
  ~ResolveEvaluator() override = default;
  MS_DECLARE_PARENT(ResolveEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list,
                         const ConfigPtr &in_conf0, const AnfNodeConfigPtr &out_conf) override {
    constexpr auto kResolveArgSize = 2;
    // Inputs: namespace, symbol
    if (args_spec_list.size() != kResolveArgSize) {
      MS_LOG(EXCEPTION) << "Expected args_spec_list size = 2, but has size:" << args_spec_list.size();
    }
    EvalResultPtr ret = nullptr;
    if (bound_node() != nullptr) {
      TraceGuard trace_guard(std::make_shared<TraceResolve>(bound_node()->debug_info()));
      ret = StaticGetter(engine, args_spec_list, in_conf0, out_conf);
    } else {
      ret = StaticGetter(engine, args_spec_list, in_conf0, out_conf);
    }
    return ret;
  }
};

bool IsContainUndetermined(const AbstractBasePtr &arg) {
  if (arg->isa<AbstractSequence>()) {
    auto seq_arg = arg->cast<AbstractSequencePtr>();
    return std::any_of(seq_arg->elements().begin(), seq_arg->elements().end(), IsContainUndetermined);
  }

  if (arg->isa<AbstractKeywordArg>()) {
    auto kw_arg = arg->cast<AbstractKeywordArgPtr>();
    return IsContainUndetermined(kw_arg->get_arg());
  }

  return arg->isa<AbstractUndetermined>() && arg->IsBroaden();
}

class CreateInstanceEvaluator : public TransitionPrimEvaluator {
 public:
  CreateInstanceEvaluator() : TransitionPrimEvaluator("CreateInstanceEvaluator") {}
  ~CreateInstanceEvaluator() override = default;
  MS_DECLARE_PARENT(CreateInstanceEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    if (args_spec_list.empty()) {
      MS_LOG(EXCEPTION) << "'args_spec_list' should not be empty";
    }

    // Get the type parameter.
    MS_EXCEPTION_IF_NULL(args_spec_list[0]);
    TypePtr type = args_spec_list[0]->GetTypeTrack();
    MS_EXCEPTION_IF_NULL(type);
    if (type->type_id() != kMetaTypeTypeType) {
      MS_LOG(EXCEPTION) << "CreateInstanceEvaluator require first parameter should be an object of TypeType, but got "
                        << type->ToString();
    }

    ValuePtr value_track = args_spec_list[0]->GetValueTrack();
    MS_EXCEPTION_IF_NULL(value_track);

    std::shared_ptr<parse::PyObjectWrapper> type_obj = dyn_cast<parse::PyObjectWrapper>(value_track);
    if (type_obj == nullptr) {
      MS_LOG(EXCEPTION) << "Cast value failed, not PyObjectWrapper:" << value_track->ToString() << ".";
    }

    if (!type_obj->isa<parse::ClassType>()) {
      MS_LOG(EXCEPTION) << "CreateInstanceEvaluator the type_obj should be an object of ClassType, but got "
                        << type_obj->ToString() << ".";
    }

    auto class_type = type_obj->obj();
    MS_LOG(DEBUG) << "Get class type is " << type_obj->ToString() << ".";

    // Get the create instance obj's parameters, `params` may contain tuple(args, kwargs).
    py::tuple params = GetParameters(args_spec_list);

    // Create class instance.
    auto obj = parse::data_converter::CreatePythonObject(class_type, params);
    if (py::isinstance<py::none>(obj)) {
      MS_LOG(EXCEPTION) << "Create python object `" << py::str(class_type)
                        << "` failed, only support to create \'Cell\' or \'Primitive\' object.";
    }

    // Process the object.
    TraceGuard guard(std::make_shared<TraceResolve>(out_conf->node()->debug_info()));
    ValuePtr converted_ret = nullptr;
    bool converted = parse::ConvertData(obj, &converted_ret, true);
    if (!converted) {
      MS_LOG(EXCEPTION) << "Convert the python object failed";
    }
    MS_EXCEPTION_IF_NULL(converted_ret);

    if (converted_ret->isa<FuncGraph>()) {
      AddToManager(engine, converted_ret->cast<FuncGraphPtr>());
    }

    AbstractBasePtr ret = ToAbstract(converted_ret, AnalysisContext::DummyContext(), out_conf);
    auto infer_result = std::make_shared<EvalResult>(ret, std::make_shared<AttrValueMap>());
    evaluator_cache_mgr_->SetValue(args_spec_list, infer_result);
    return infer_result;
  }

  py::tuple GetParameters(const AbstractBasePtrList &args_spec_list) const {
    if (args_spec_list.empty()) {
      MS_LOG(EXCEPTION) << "Unexpected arguments num, the min arguments num must be 1, but got 0.";
    }
    // Exclude class type by minus 1;
    std::size_t params_size = args_spec_list.size() - 1;
    auto params = py::tuple(params_size);
    for (size_t i = 0; i < params_size; i++) {
      // Only support the Scalar parameters type. Bypass class type by offset with 1.
      auto arg = args_spec_list[i + 1];
      MS_EXCEPTION_IF_NULL(arg);
      if (IsContainUndetermined(arg)) {
        MS_EXCEPTION(TypeError) << "The " << i << "th input of method __init__ for "
                                << args_spec_list[0]->BuildValue()->ToString()
                                << " should be a scalar but got:" << arg->ToString();
      }
      // Because the Tensor's AbstractTensor can't get value from GetValueTrack.
      ValuePtr param_value = arg->BuildValue();
      py::object param = ValueToPyData(param_value);
      params[i] = param;
    }
    return params;
  }
};

class PyInterpretEvaluator : public TransitionPrimEvaluator {
 public:
  PyInterpretEvaluator() : TransitionPrimEvaluator("PyInterpretEvaluator") {}
  ~PyInterpretEvaluator() override = default;
  MS_DECLARE_PARENT(PyInterpretEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    if (args_spec_list.empty()) {
      MS_LOG(ERROR) << "'args_spec_list' should not be empty";
    }

    // Get the type parameter.
    MS_EXCEPTION_IF_NULL(args_spec_list[0]);
    ValuePtr value_track = args_spec_list[0]->GetValueTrack();
    MS_EXCEPTION_IF_NULL(value_track);

    std::shared_ptr<parse::Script> script_obj = dyn_cast<parse::Script>(value_track);
    if (script_obj == nullptr) {
      MS_LOG(EXCEPTION) << "Cast value failed, not PyObjectWrapper:" << value_track->ToString() << ".";
    }

    // Make global and local parameters.
    py::tuple params = MakeParameters(args_spec_list);

    // Call python script string.
    MS_LOG(DEBUG) << "Call script: " << script_obj->script() << ", params: " << py::str(params);
    auto obj = parse::data_converter::CallPythonScript(py::str(script_obj->script()), params);
    if (py::isinstance<py::none>(obj)) {
      AbstractBasePtr res = std::make_shared<abstract::AbstractNone>();
      auto infer_result = std::make_shared<EvalResult>(res, nullptr);
      evaluator_cache_mgr_->SetValue(args_spec_list, infer_result);
      return infer_result;
    }

    ValuePtr converted_val = nullptr;
    bool converted = parse::ConvertData(obj, &converted_val, true);
    if (!converted) {
      MS_LOG(EXCEPTION) << "Convert the python object failed";
    }
    MS_EXCEPTION_IF_NULL(converted_val);

    AbstractBasePtr res = ToAbstract(converted_val, AnalysisContext::DummyContext(), out_conf);
    auto infer_result = std::make_shared<EvalResult>(res, std::make_shared<AttrValueMap>());
    evaluator_cache_mgr_->SetValue(args_spec_list, infer_result);
    return infer_result;
  }

  py::tuple MakeParameters(const AbstractBasePtrList &args_spec_list) const {
    constexpr int params_size = 3;
    if (params_size != args_spec_list.size()) {
      MS_LOG(EXCEPTION) << "Unexpected params_size: " << params_size
                        << ", not equal to arguments.size:" << args_spec_list.size();
    }
    // The first argument is script string, ignore it.
    auto params = py::tuple(params_size - 1);

    // Make the global parameters.
    auto global_dict = dyn_cast<AbstractDictionary>(args_spec_list[1]);  // Global parameters dict.
    MS_EXCEPTION_IF_NULL(global_dict);
    auto filtered_global_dict = FilterParameters(global_dict);
    MS_LOG(DEBUG) << "arg_1, global_dict: " << global_dict->ToString()
                  << ", filtered_global_dict: " << filtered_global_dict->ToString();
    ValuePtr global_dict_value = filtered_global_dict->BuildValue();
    py::object global_params_dict = ValueToPyData(global_dict_value);
    MS_LOG(DEBUG) << "arg_1, python global_params_dict: " << global_dict_value->ToString() << " -> "
                  << py::str(global_params_dict);
    params[0] = global_params_dict;

    // Make the local parameters.
    constexpr size_t local_index = 2;
    auto local_dict = dyn_cast<AbstractDictionary>(args_spec_list[local_index]);  // Local parameters dict.
    MS_EXCEPTION_IF_NULL(local_dict);
    auto filtered_local_dict = FilterParameters(local_dict);
    MS_LOG(DEBUG) << "arg_2, local_dict: " << local_dict->ToString()
                  << ", filtered_local_dict:" << filtered_local_dict->ToString();
    ValuePtr local_dict_value = filtered_local_dict->BuildValue();
    py::dict local_params_dict = ReCheckLocalDict(filtered_local_dict);
    MS_LOG(DEBUG) << "arg_2, python local_params_dict: " << local_dict_value->ToString() << " -> "
                  << py::str(local_params_dict);
    params[1] = local_params_dict;
    return params;
  }

  py::dict ReCheckLocalDict(const AbstractDictionaryPtr &filtered_local_dict) const {
    const auto &keys_values = filtered_local_dict->elements();
    py::dict local_params_dict;
    for (auto &key_value : keys_values) {
      ValuePtr element_value = key_value.second->BuildValue();
      MS_EXCEPTION_IF_NULL(element_value);
      auto py_data = ValueToPyData(element_value);
      local_params_dict[py::str(key_value.first)] = py_data;
    }
    return local_params_dict;
  }

  AbstractDictionaryPtr FilterParameters(const AbstractDictionaryPtr &abstract_dict) const {
    std::vector<AbstractAttribute> kv;
    const auto &keys_values = abstract_dict->elements();
    // Filter out the element of Function type.
    (void)std::copy_if(keys_values.cbegin(), keys_values.cend(), std::back_inserter(kv),
                       [](const AbstractAttribute &item) {
                         MS_EXCEPTION_IF_NULL(item.second);
                         return (!item.second->isa<abstract::AbstractFunction>());
                       });
    return std::make_shared<AbstractDictionary>(kv);
  }
};

class MakeTupleEvaluator : public TransitionPrimEvaluator {
 public:
  MakeTupleEvaluator() : TransitionPrimEvaluator("MakeTupleEvaluator") {}
  ~MakeTupleEvaluator() override = default;
  MS_DECLARE_PARENT(MakeTupleEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    if (args_spec_list.empty()) {
      MS_LOG(INFO) << "For MakeTuple, the inputs should not be empty. node: " << out_conf->node()->DebugString();
    }

    static const auto enable_eliminate_unused_element = (common::GetEnv("MS_DEV_ENABLE_DDE") != "0");
    if (enable_eliminate_unused_element) {
      auto flags = GetSequenceNodeElementsUseFlags(out_conf->node());
      if (flags == nullptr) {
        SetSequenceNodeElementsUseFlags(out_conf->node(), std::make_shared<std::vector<bool>>(args_spec_list.size()));
      }
    }
    std::shared_ptr<AnfNodeWeakPtrList> sequence_nodes = std::make_shared<AnfNodeWeakPtrList>();
    if (enable_eliminate_unused_element) {
      (void)sequence_nodes->emplace_back(AnfNodeWeakPtr(out_conf->node()));
    }
    auto abs = std::make_shared<AbstractTuple>(args_spec_list, sequence_nodes);
    auto res = std::make_shared<EvalResult>(abs, std::make_shared<AttrValueMap>());
    evaluator_cache_mgr_->SetValue(args_spec_list, res);
    return res;
  }
};

class MakeListEvaluator : public TransitionPrimEvaluator {
 public:
  MakeListEvaluator() : TransitionPrimEvaluator("MakeListEvaluator") {}
  ~MakeListEvaluator() override = default;
  MS_DECLARE_PARENT(MakeListEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    if (args_spec_list.empty()) {
      MS_LOG(INFO) << "For MakeList, the inputs should not be empty. node: " << out_conf->node()->DebugString();
    }

    static const auto enable_eliminate_unused_element = (common::GetEnv("MS_DEV_ENABLE_DDE") != "0");
    if (enable_eliminate_unused_element) {
      auto flags = GetSequenceNodeElementsUseFlags(out_conf->node());
      if (flags == nullptr) {
        SetSequenceNodeElementsUseFlags(out_conf->node(), std::make_shared<std::vector<bool>>(args_spec_list.size()));
      }
    }
    std::shared_ptr<AnfNodeWeakPtrList> sequence_nodes = std::make_shared<AnfNodeWeakPtrList>();
    if (enable_eliminate_unused_element) {
      (void)sequence_nodes->emplace_back(AnfNodeWeakPtr(out_conf->node()));
    }
    auto abs = std::make_shared<AbstractList>(args_spec_list, sequence_nodes);
    auto res = std::make_shared<EvalResult>(abs, std::make_shared<AttrValueMap>());
    evaluator_cache_mgr_->SetValue(args_spec_list, res);
    return res;
  }
};

class PartialEvaluator : public Evaluator {
 public:
  PartialEvaluator() : Evaluator("PartialEvaluator") {}
  ~PartialEvaluator() override = default;
  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                    const AnfNodeConfigPtr &out_conf) override {
    if (args_conf_list.size() == 0) {
      MS_LOG(EXCEPTION) << "Args size should be greater than 0";
    }

    MS_EXCEPTION_IF_NULL(out_conf);
    MS_EXCEPTION_IF_NULL(out_conf->node());
    MS_EXCEPTION_IF_NULL(args_conf_list[0]);
    MS_EXCEPTION_IF_NULL(args_conf_list[0]->ObtainEvalResult());
    auto arg0_value = args_conf_list[0]->ObtainEvalResult()->abstract();
    MS_EXCEPTION_IF_NULL(arg0_value);
    AbstractBasePtrList args_spec_list{arg0_value};
    // Func in hypermap(partial(Func, arg0), arg1, arg2) may become Poly Node.
    if (arg0_value->isa<AbstractError>()) {
      MS_EXCEPTION_IF_NULL(arg0_value->GetValueTrack());
      auto ret = std::make_shared<AbstractError>(arg0_value->GetValueTrack()->cast<StringImmPtr>(), out_conf->node());
      MS_LOG(DEBUG) << "AbstractError for node: " << out_conf->node()->DebugString()
                    << " as func is: " << arg0_value->ToString();
      auto eval_result = std::make_shared<EvalResult>(ret, std::make_shared<AttrValueMap>());
      evaluator_cache_mgr_->SetValue(args_spec_list, eval_result);
      return eval_result;
    }
    auto func = CheckArg<AbstractFunction>("partial", args_spec_list, 0);
    // Sometimes, node[0] in out_conf becomes phi0;
    if (func->isa<PrimitiveAbstractClosure>()) {
      auto prim_func = dyn_cast<PrimitiveAbstractClosure>(func);
      MS_EXCEPTION_IF_NULL(prim_func->prim());
      if (prim_func->prim()->isa<prim::DoSignaturePrimitive>()) {
        prim::DoSignaturePrimitivePtr do_signature_prim = dyn_cast<prim::DoSignaturePrimitive>(prim_func->prim());
        return HandleDoSignature(engine, do_signature_prim->function(), out_conf);
      }
    }

    (void)std::transform(
      args_conf_list.begin() + 1, args_conf_list.end(), std::back_inserter(args_spec_list),
      [](const ConfigPtr &config) -> AbstractBasePtr { return config->ObtainEvalResult()->abstract(); });
    AbstractBasePtrList args(args_spec_list.begin() + 1, args_spec_list.end());

    auto cnode = out_conf->node()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->size() != (args_conf_list.size() + 1)) {
      MS_LOG(EXCEPTION) << "Out_conf node: " << cnode->DebugString()
                        << ", args_conf_list: " << mindspore::ToString(args_conf_list);
    }
    AbstractFuncAtomPtrList partial_funcs_list;
    auto build_partial = [args, cnode, &partial_funcs_list](const AbstractFuncAtomPtr &atom_func) {
      auto new_func = std::make_shared<PartialAbstractClosure>(atom_func, args, cnode);
      partial_funcs_list.push_back(new_func);
    };
    func->Visit(build_partial);

    auto ret = AbstractFunction::MakeAbstractFunction(partial_funcs_list);
    auto eval_result = std::make_shared<EvalResult>(ret, std::make_shared<AttrValueMap>());
    evaluator_cache_mgr_->SetValue(args_spec_list, eval_result);
    return eval_result;
  }

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &, const AnfNodeConfigPtr &) override {
    MS_LOG(EXCEPTION) << "Eval() should not be called, Run() method should be called";
  }

  EvalResultPtr HandleDoSignature(const AnalysisEnginePtr &engine, const ValuePtr &signature_value,
                                  const AnfNodeConfigPtr &out_conf) const {
    MS_EXCEPTION_IF_NULL(engine);
    MS_EXCEPTION_IF_NULL(out_conf);
    MS_EXCEPTION_IF_NULL(out_conf->node());
    auto cnode = out_conf->node()->cast<CNodePtr>();
    if (cnode == nullptr) {
      MS_LOG(EXCEPTION) << "Cnode is nullptr";
    }

    ScopeGuard scope_guard(out_conf->node()->scope());
    TraceGuard trace_guard(std::make_shared<TraceDoSignature>(out_conf->node()->debug_info()));
    std::vector<AnfNodePtr> new_nodes_inputs = cnode->inputs();
    auto new_signature_value = std::make_shared<prim::DoSignatureMetaFuncGraph>("signature", signature_value);
    new_nodes_inputs[1] = NewValueNode(new_signature_value);
    FuncGraphPtr func_graph = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    CNodePtr new_cnode = func_graph->NewCNode(std::move(new_nodes_inputs));
    AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_cnode, out_conf->context(), out_conf->func_graph());
    return engine->ForwardConfig(out_conf, fn_conf);
  }
};

class RaiseEvaluator : public TransitionPrimEvaluator {
 public:
  RaiseEvaluator() : TransitionPrimEvaluator("RaiseEvaluator") {}
  ~RaiseEvaluator() override = default;
  MS_DECLARE_PARENT(RaiseEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list,
                         const ConfigPtr &in_conf0, const AnfNodeConfigPtr &out_conf) override {
    auto node = out_conf->node();
    MS_EXCEPTION_IF_NULL(node);
    auto cur_graph = node->func_graph();
    MS_EXCEPTION_IF_NULL(cur_graph);
    if (cur_graph->is_tensor_condition_branch()) {
      MS_LOG(EXCEPTION) << "Currently only supports raise in constant scenarios."
                        << "Tensor type data cannot exist in the conditional statement."
                        << "Please check your conditions which raise node is located at:"
                        << trace::GetDebugInfo(node->debug_info()) << ".";
    }
    if (args_spec_list.empty()) {
      // process raise
      MS_LOG(EXCEPTION) << "No active exception to reraise.";
    }

    std::string exception_type = GetScalarStringValue(args_spec_list[0]);
    auto iter = exception_types_map.find(exception_type);
    if (iter == exception_types_map.end()) {
      MS_LOG(EXCEPTION) << "Unsupported exception type: " << exception_type << ".";
    }
    ExceptionType type = iter->second;
    if (args_spec_list.size() == 1) {
      // Process raise ValueError()
      MS_EXCEPTION(type);
    }
    std::string exception_string = "";
    for (size_t index = 1; index < args_spec_list.size(); ++index) {
      exception_string += GetExceptionString(args_spec_list[index]);
    }
    MS_EXCEPTION(type) << exception_string;
    return nullptr;
  }

 private:
  std::string GetExceptionString(const AbstractBasePtr &arg) {
    std::string exception_str = "";
    if (arg->isa<abstract::AbstractTuple>()) {
      // Process raise ValueError("str")
      auto arg_tuple = arg->cast<abstract::AbstractTuplePtr>();
      const auto &arg_tuple_elements = arg_tuple->elements();
      if (arg_tuple_elements.size() == 0) {
        MS_LOG(EXCEPTION) << "The arg_tuple_elements can't be empty.";
      }
      for (size_t index = 0; index < arg_tuple_elements.size(); ++index) {
        auto &element = arg_tuple_elements[index];
        exception_str += GetScalarStringValue(element);
      }
    } else {
      // Process raise ValueError
      exception_str += GetScalarStringValue(arg);
    }
    return exception_str;
  }

  std::string GetScalarStringValue(const AbstractBasePtr &abs) {
    std::string str = "";
    if (abs->isa<abstract::AbstractScalar>()) {
      auto scalar = abs->cast<abstract::AbstractScalarPtr>();
      auto scalar_value = scalar->BuildValue();
      if (scalar_value->isa<Int64Imm>()) {
        str = std::to_string(GetValue<int64_t>(scalar_value));
      } else if (scalar_value->isa<StringImm>()) {
        str = GetValue<std::string>(scalar_value);
      }
    }
    return str;
  }
};

struct PrimitiveImplInferValue {
  PrimitiveImpl impl_;        // implement function of primitive
  bool eval_value_;           // whether evaluate value
  TypePtr specify_out_type_;  // whether specify return type
  bool in_white_list_;        // true if this Primitive in white list, else false.
};

using PrimitiveToImplMap = mindspore::HashMap<PrimitivePtr, PrimitiveImplInferValue, PrimitiveHasher, PrimitiveEqual>;
PrimitiveToImplMap &GetUniformPrimitiveToImplMap() {
  using R = PrimitiveToImplMap::mapped_type;
  static PrimitiveToImplMap uniform_prim_implement_map{
    {prim::kPrimScalarAdd, R{prim::ScalarAdd, true, nullptr, true}},
    {prim::kPrimScalarSub, R{prim::ScalarSub, true, nullptr, true}},
    {prim::kPrimScalarMul, R{prim::ScalarMul, true, nullptr, true}},
    {prim::kPrimScalarDiv, R{prim::ScalarDiv, true, nullptr, true}},
    {prim::kPrimScalarMod, R{prim::ScalarMod, true, nullptr, true}},
    {prim::kPrimScalarPow, R{prim::ScalarPow, true, nullptr, true}},
    {prim::kPrimScalarFloordiv, R{prim::ScalarFloordiv, true, nullptr, true}},
    {prim::kPrimScalarUadd, R{prim::ScalarUAdd, true, nullptr, true}},
    {prim::kPrimScalarUsub, R{prim::ScalarUSub, true, nullptr, true}},
    {prim::kPrimScalarLog, R{prim::ScalarLog, true, nullptr, true}},
    {prim::kPrimScalarEq, R{prim::ScalarEq, true, std::make_shared<Bool>(), true}},
    {prim::kPrimScalarLt, R{prim::ScalarLt, true, std::make_shared<Bool>(), true}},
    {prim::kPrimScalarGt, R{prim::ScalarGt, true, std::make_shared<Bool>(), true}},
    {prim::kPrimScalarNe, R{prim::ScalarNe, true, std::make_shared<Bool>(), true}},
    {prim::kPrimScalarLe, R{prim::ScalarLe, true, std::make_shared<Bool>(), true}},
    {prim::kPrimScalarGe, R{prim::ScalarGe, true, std::make_shared<Bool>(), true}},
    {prim::kPrimBoolNot, R{prim::BoolNot, true, std::make_shared<Bool>(), true}},
    {prim::kPrimBoolAnd, R{prim::BoolAnd, true, std::make_shared<Bool>(), true}},
    {prim::kPrimBoolEq, R{prim::BoolEq, true, std::make_shared<Bool>(), true}},
    {prim::kPrimBoolOr, R{prim::BoolOr, true, std::make_shared<Bool>(), true}},
  };
  return uniform_prim_implement_map;
}

PrimEvaluatorMap PrimEvaluatorConstructors = PrimEvaluatorMap();
std::mutex PrimEvaluatorConstructorMutex;

void InitPrimEvaluatorConstructors() {
  PrimEvaluatorMap &constructor = PrimEvaluatorConstructors;

  for (const auto &iter : GetPrimitiveToEvalImplMap()) {
    constructor[iter.first] = InitStandardPrimEvaluator(iter.first, iter.second);
  }

  for (const auto &iter : GetUniformPrimitiveToImplMap()) {
    constructor[iter.first] =
      InitUniformPrimEvaluator(iter.first, iter.second.impl_, iter.second.eval_value_, iter.second.specify_out_type_);
  }
  constructor[prim::kPrimEmbed] = std::make_shared<EmbedEvaluator>();
  constructor[prim::kPrimRefToEmbed] = std::make_shared<RefToEmbedEvaluator>();
  constructor[prim::kPrimGetAttr] = std::make_shared<GetAttrEvaluator>();
  constructor[prim::kPrimResolve] = std::make_shared<ResolveEvaluator>();
  constructor[prim::kPrimCreateInstance] = std::make_shared<CreateInstanceEvaluator>();
  constructor[prim::kPrimPartial] = std::make_shared<PartialEvaluator>();
  constructor[prim::kPrimPyInterpret] = std::make_shared<PyInterpretEvaluator>();
  constructor[prim::kPrimMakeTuple] = std::make_shared<MakeTupleEvaluator>();
  constructor[prim::kPrimMakeList] = std::make_shared<MakeListEvaluator>();
  constructor[prim::kPrimRaise] = std::make_shared<RaiseEvaluator>();
}
}  // namespace

void ClearPrimEvaluatorMap() {
  PrimEvaluatorConstructors.clear();
  GetPrimitiveToEvalImplMap().clear();
  GetUniformPrimitiveToImplMap().clear();
}

bool IsInWhiteList(const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(primitive);

  auto iter = GetPrimitiveToEvalImplMap().find(primitive);
  if (iter != GetPrimitiveToEvalImplMap().end()) {
    return iter->second.in_white_list_;
  }

  auto uni_iter = GetUniformPrimitiveToImplMap().find(primitive);
  if (uni_iter != GetUniformPrimitiveToImplMap().end()) {
    return uni_iter->second.in_white_list_;
  }

  return false;
}

PrimEvaluatorMap &GetPrimEvaluatorConstructors() {
  PrimEvaluatorMap &constructor = PrimEvaluatorConstructors;
  if (!constructor.empty()) {
    return constructor;
  }
  std::lock_guard<std::mutex> initLock(PrimEvaluatorConstructorMutex);
  if (constructor.empty()) {
    InitPrimEvaluatorConstructors();
  }

  return constructor;
}

namespace {
bool IsSubtypeTuple(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  auto x_tuple = dyn_cast<AbstractTuple>(x);
  auto model_tuple = dyn_cast<Tuple>(model);

  if (x_tuple == nullptr || model_tuple == nullptr) {
    return false;
  }

  if (model->IsGeneric()) {
    return true;
  }

  if (x_tuple->size() != model_tuple->size()) {
    return false;
  }

  for (size_t i = 0; i < x_tuple->size(); i++) {
    bool is_subtype = IsSubtype((*x_tuple)[i], (*model_tuple)[i]);
    if (!is_subtype) {
      return false;
    }
  }
  return true;
}

bool IsSubtypeArray(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  auto x_tensor = dyn_cast<AbstractTensor>(x);
  auto model_tensor = dyn_cast<TensorType>(model);

  if (x_tensor == nullptr || model_tensor == nullptr) {
    return false;
  }

  if (model->IsGeneric()) {
    return true;
  }

  return IsSubtype(x_tensor->element(), model_tensor->element());
}

bool IsSubtypeList(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  auto x_list = dyn_cast<AbstractList>(x);
  auto model_list = dyn_cast<List>(model);

  if (x_list == nullptr || model_list == nullptr) {
    return false;
  }

  if (model->IsGeneric()) {
    return true;
  }

  if (x_list->size() != model_list->size()) {
    return false;
  }

  bool is_subtype = true;
  for (size_t i = 0; i < x_list->size(); i++) {
    is_subtype = IsSubtype((*x_list)[i], (*model_list)[i]);
    if (!is_subtype) {
      return false;
    }
  }
  return is_subtype;
}

bool IsSubtypeClass(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  auto x_class = dyn_cast<AbstractClass>(x);
  auto model_class = dyn_cast<Class>(model);
  if (x_class == nullptr) {
    return false;
  }
  if (model->IsGeneric()) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(model_class);
  if (x_class->tag() == model_class->tag()) {
    auto m_attributes = model_class->GetAttributes();
    auto x_attributes = x_class->attributes();
    if (m_attributes.size() != x_attributes.size()) {
      return false;
    }

    for (size_t i = 0; i < m_attributes.size(); i++) {
      if (!IsSubtype(x_attributes[i].second, m_attributes[i].second)) {
        return false;
      }
    }
    return true;
  }

  return false;
}

inline bool IsSubtypeScalar(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  if (dyn_cast<AbstractScalar>(x) == nullptr) {
    return false;
  }
  TypePtr x_type = x->GetTypeTrack();
  return IsSubType(x_type, model);
}
}  // namespace

bool IsSubtype(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  TypeId model_typeid = model->type_id();
  switch (model_typeid) {
    case kMetaTypeObject:
      return true;
    case kObjectTypeTuple:
      return IsSubtypeTuple(x, model);
    case kObjectTypeTensorType:
      return IsSubtypeArray(x, model);
    case kObjectTypeList:
      return IsSubtypeList(x, model);
    case kObjectTypeClass:
      return IsSubtypeClass(x, model);
    default:
      if (IsSubType(model, std::make_shared<Number>())) {
        return IsSubtypeScalar(x, model);
      }
      MS_LOG(EXCEPTION) << "Invalid model type: " << model->ToString() << ".";
  }
}
}  // namespace abstract
}  // namespace mindspore
