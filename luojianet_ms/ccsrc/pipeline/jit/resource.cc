/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/resource.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
#include "pipeline/jit/debug/trace.h"
#include "ir/dtype.h"
#include "pipeline/jit/parse/data_converter.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/ad/dfunctor.h"
#include "include/common/utils/parallel_context.h"

namespace luojianet_ms {
// namespace to support opmap definition
namespace pipeline {

BuiltInTypeMap &GetMethodMap() {
  static BuiltInTypeMap method_map = {{kObjectTypeString,
                                       {
                                         {"__bool__", std::string("str_bool")}  // C.str_bool
                                       }},
                                      {kMetaTypeNone,
                                       {
                                         {"__bool__", std::string("none_bool")}  // C.none_bool
                                       }},
                                      {kObjectTypeFunction,
                                       {{"__bool__", std::string("func_bool")},  // C.str_bool
                                        {"__is_csr_func__", prim::kPrimIsCSRFunc}}},
                                      {kNumberTypeBool,
                                       {
                                         {"__and__", prim::kPrimBoolAnd},     // P.bool_and
                                         {"__or__", prim::kPrimBoolOr},       // P.bool_or
                                         {"__eq__", prim::kPrimBoolEq},       // P.bool_eq
                                         {"__ne__", std::string("bool_ne")},  // C.bool_ne
                                         {"__bool__", prim::kPrimIdentity}    // P.identity
                                       }},
                                      {kNumberTypeInt,
                                       {
                                         {"__add__", prim::kPrimScalarAdd},              // P.scalar_add
                                         {"__sub__", prim::kPrimScalarSub},              // P.scalar_sub
                                         {"__mul__", prim::kPrimScalarMul},              // P.scalar_mul
                                         {"__floordiv__", std::string("int_floordiv")},  // C.int_floordiv
                                         {"__truediv__", std::string("int_truediv")},    // C.int_truediv
                                         {"__mod__", prim::kPrimScalarMod},              // P.scalar_mod
                                         {"__pow__", prim::kPrimScalarPow},              // P.scalar_pow
                                         {"__floor__", prim::kPrimIdentity},             // P.identity
                                         {"__trunc__", prim::kPrimIdentity},             // P.identity
                                         {"__pos__", prim::kPrimScalarUadd},             // P.scalar_uadd
                                         {"__neg__", prim::kPrimScalarUsub},             // P.scalar_usub
                                         {"__eq__", prim::kPrimScalarEq},                // P.scalar_eq
                                         {"__ne__", prim::kPrimScalarNe},                // P.scalar_ne
                                         {"__lt__", prim::kPrimScalarLt},                // P.scalar_lt
                                         {"__gt__", prim::kPrimScalarGt},                // P.scalar_gt
                                         {"__le__", prim::kPrimScalarLe},                // P.scalar_le
                                         {"__ge__", prim::kPrimScalarGe},                // P.scalar_ge
                                         {"__bool__", std::string("int_bool")},          // C.int_bool
                                         {"__ms_to_array__", prim::kPrimScalarToArray},  // P.scalar_to_array
                                       }},
                                      {kNumberTypeUInt,
                                       {
                                         {"__add__", prim::kPrimScalarAdd},              // P.scalar_add,
                                         {"__sub__", prim::kPrimScalarSub},              // P.scalar_sub,
                                         {"__mul__", prim::kPrimScalarMul},              // P.scalar_mul,
                                         {"__floordiv__", prim::kPrimScalarDiv},         // P.scalar_div,
                                         {"__truediv__", std::string("int_truediv")},    // C.int_truediv
                                         {"__mod__", prim::kPrimScalarMod},              // P.scalar_mod,
                                         {"__pow__", prim::kPrimScalarPow},              // P.scalar_pow,
                                         {"__floor__", prim::kPrimIdentity},             // P.identity,
                                         {"__trunc__", prim::kPrimIdentity},             // P.identity,
                                         {"__pos__", prim::kPrimScalarUadd},             // P.scalar_uadd,
                                         {"__neg__", prim::kPrimScalarUsub},             // P.scalar_usub,
                                         {"__eq__", prim::kPrimScalarEq},                // P.scalar_eq,
                                         {"__ne__", prim::kPrimScalarNe},                // P.scalar_ne,
                                         {"__lt__", prim::kPrimScalarLt},                // P.scalar_lt,
                                         {"__gt__", prim::kPrimScalarGt},                // P.scalar_gt,
                                         {"__le__", prim::kPrimScalarLe},                // P.scalar_le,
                                         {"__ge__", prim::kPrimScalarGe},                // P.scalar_ge,
                                         {"__bool__", std::string("int_bool")},          // C.int_bool
                                         {"__ms_to_array__", prim::kPrimScalarToArray},  // P.scalar_to_array,
                                       }},
                                      {kNumberTypeFloat,
                                       {
                                         {"__add__", prim::kPrimScalarAdd},                // P.scalar_add,
                                         {"__sub__", prim::kPrimScalarSub},                // P.scalar_sub,
                                         {"__mul__", prim::kPrimScalarMul},                // P.scalar_mul,
                                         {"__floordiv__", std::string("float_floordiv")},  // C.float_floordiv
                                         {"__truediv__", prim::kPrimScalarDiv},            // P.scalar_div,
                                         {"__mod__", prim::kPrimScalarMod},                // P.scalar_mod,
                                         {"__pow__", prim::kPrimScalarPow},                // P.scalar_pow,
                                         {"__floor__", prim::kPrimScalarFloor},            // P.scalar_floor,
                                         {"__trunc__", prim::kPrimScalarTrunc},            // P.scalar_trunc,
                                         {"__pos__", prim::kPrimScalarUadd},               // P.scalar_uadd,
                                         {"__neg__", prim::kPrimScalarUsub},               // P.scalar_usub,
                                         {"__eq__", prim::kPrimScalarEq},                  // P.scalar_eq,
                                         {"__ne__", prim::kPrimScalarNe},                  // P.scalar_ne,
                                         {"__lt__", prim::kPrimScalarLt},                  // P.scalar_lt,
                                         {"__gt__", prim::kPrimScalarGt},                  // P.scalar_gt,
                                         {"__le__", prim::kPrimScalarLe},                  // P.scalar_le,
                                         {"__ge__", prim::kPrimScalarGe},                  // P.scalar_ge,
                                         {"__bool__", std::string("float_bool")},          // C.float_bool
                                         {"__ms_to_array__", prim::kPrimScalarToArray},    // P.scalar_to_array,
                                       }},
                                      {kObjectTypeTuple,
                                       {
                                         {"__len__", prim::kPrimTupleLen},                  // P.tuple_len,
                                         {"__getitem__", prim::kPrimTupleGetItem},          // P.tuple_getitem,
                                         {"__setitem__", prim::kPrimTupleSetItem},          // P.tuple_setitem,
                                         {"__ms_iter__", prim::kPrimIdentity},              // P.identity,
                                         {"__ms_next__", std::string("tuple_next")},        // C.tuple_next,
                                         {"__ms_hasnext__", std::string("tuple_hasnext")},  // C.tuple_hasnext
                                         {"__bool__", std::string("tuple_bool")}            // C.tuple_bool
                                       }},
                                      {kObjectTypeList,
                                       {
                                         {"__len__", prim::kPrimListLen},            // P.list_len,
                                         {"__getitem__", prim::kPrimListGetItem},    // P.list_getitem,
                                         {"__setitem__", prim::kPrimListSetItem},    // P.list_setitem,
                                         {"__ms_iter__", prim::kPrimIdentity},       // P.identity
                                         {"__ms_next__", std::string("list_next")},  // C.list_next
                                         {"append", std::string("list_append")},     // C.list_next
                                         {"__bool__", std::string("list_bool")},     // C.list_bool
                                         {"__ms_hasnext__", std::string("list_hasnext")},
                                         {"insert", std::string("list_insert")},
                                       }},
                                      {kObjectTypeDictionary,
                                       {
                                         {"__len__", prim::kPrimDictLen},          // P.dict_len
                                         {"__getitem__", prim::kPrimDictGetItem},  // P.dict_getitem
                                         {"__setitem__", prim::kPrimDictSetItem},  // P.dict_setitem,
                                         {"keys", prim::kPrimDictGetKeys},         // P.dict_getkeys,
                                         {"values", prim::kPrimDictGetValues},     // P.dict_getvalues,
                                         {"items", prim::kPrimDictItems},          // P.dict_items
                                         {"__bool__", std::string("dict_bool")}    // C.dict_bool
                                       }},
                                      {kObjectTypeTensorType,
                                       {
                                         {"all", std::string("all_")},                    // C.reduce_all
                                         {"any", std::string("any_")},                    // C.reduce_any
                                         {"__add__", std::string("add")},                 // C.add
                                         {"__sub__", std::string("sub")},                 // C.sub
                                         {"__mul__", std::string("mul")},                 // C.mul
                                         {"abs", std::string("abs_")},                    // C.abs_
                                         {"mean", std::string("mean")},                   // C.mean
                                         {"__truediv__", std::string("truediv")},         // C.truediv
                                         {"__floordiv__", std::string("floordiv")},       // C.floordiv
                                         {"__mod__", std::string("mod")},                 // C.mod
                                         {"__pow__", std::string("pow_")},                // C.pow
                                         {"__floor__", std::string("array_floor")},       // C.array_floor
                                         {"__trunc__", std::string("array_trunc")},       // C.array_trunc
                                         {"__pos__", std::string("array_uadd")},          // C.array_uadd
                                         {"__neg__", std::string("array_usub")},          // C.array_usub
                                         {"__eq__", std::string("eq")},                   // C.eq
                                         {"__ne__", std::string("ne")},                   // C.ne
                                         {"__lt__", std::string("lt")},                   // C.lt
                                         {"__gt__", std::string("gt")},                   // C.gt
                                         {"__le__", std::string("le")},                   // C.le
                                         {"__ge__", std::string("ge")},                   // C.ge
                                         {"expand_as", std::string("expand_tensor_as")},  // C.expand_as
                                         {"view", std::string("view")},                   // C.view
                                         {"__len__", prim::kPrimArrayLen},                // P.array_len,
                                         {"__getitem__", prim::kPrimArrayGetItem},        // P.array_getitem,
                                         {"__setitem__", prim::kPrimArraySetItem},        // P.array_setitem,
                                         {"__ms_iter__", std::string("array_iter")},      // C.array_iter
                                         {"__ms_to_array__", prim::kPrimIdentity},        // P.identity,
                                         {"item", std::string("item")},                   // P.item,
                                         {"itemset", std::string("itemset")},             // P.itemset,
                                         {"transpose", std::string("transpose")},         // P.transpose
                                         {"flatten", std::string("flatten")},             // P.reshape(,-1)
                                         {"reshape", std::string("reshape")},             // P.reshape()
                                         {"ravel", std::string("ravel")},                 // P.reshape(,(-1,))
                                         {"swapaxes", std::string("swapaxes")},           // P.transpose()
                                         {"narrow", std::string("narrow")},               // narrow()
                                         {"masked_fill", std::string("masked_fill")},     // masked_fill()
                                         {"expand_dims", std::string("expand_dims")},     // P.expand_dims()
                                         {"squeeze", std::string("squeeze")},             // P.squeeze()
                                         {"astype", std::string("astype")},               // P.cast()
                                         {"cumsum", std::string("cumsum")},               // P.cumsum()
                                         {"copy", std::string("copy")},                   // copy()
                                         {"max", std::string("max")},                     // P.reduce_max()
                                         {"min", std::string("min")},                     // P.reduce_min()
                                         {"fill", std::string("fill")},                   // P.fill()
                                         {"ptp", std::string("ptp")},               // P.reduce_max() - P.reduce_min()
                                         {"clip", std::string("clip")},             // P.maximum(P.minimum)
                                         {"__bool__", std::string("tensor_bool")},  // C.tensor_bool
                                         {"argmax", std::string("argmax")},         // P.Argmax()
                                         {"argmin", std::string("argmin")},         // P.Argmax()
                                         {"resize", std::string("resize")},         // P.Reshape()
                                         {"choose", std::string("choose")},         // P.Select()
                                         {"diagonal", std::string("diagonal")},     // P.Eye()
                                         {"searchsorted", std::string("searchsorted")},  // P.Select()
                                         {"take", std::string("take")},                  // P.GatherNd()
                                         {"trace", std::string("trace")},                // P.Eye()
                                         {"var", std::string("var")},                    // P.ReduceSum
                                         {"std", std::string("std")},                    // P.ReduceSum
                                         {"sum", std::string("sum")},                    // P.ReduceSum
                                         {"repeat", std::string("repeat")},              // C.repeat_elements
                                       }},
                                      {kObjectTypeRowTensorType,
                                       {
                                         {"__add__", prim::kPrimRowTensorAdd},  // P.row_tensor_add
                                       }},
                                      {kObjectTypeCSRTensorType,
                                       {
                                         {"astype", std::string("csr_astype")},      // C.csr_astype
                                         {"abs", std::string("csr_abs")},            // C.csr_abs
                                         {"sum", std::string("csr_sum")},            // C.csr_sum
                                         {"mv", std::string("csr_mv")},              // C.csr_mv
                                         {"to_tuple", std::string("csr_to_tuple")},  // C.csr_to_tuple
                                         {"to_coo", std::string("csr_to_coo")},      // C.csr_to_coo
                                         {"to_dense", std::string("csr_to_dense")},  // C.csr_to_dense
                                       }},
                                      {kObjectTypeCOOTensorType,
                                       {
                                         {"astype", std::string("coo_astype")},      // C.coo_astype
                                         {"abs", std::string("coo_abs")},            // C.coo_abs
                                         {"to_tuple", std::string("coo_to_tuple")},  // C.coo_to_tuple
                                         {"to_csr", std::string("coo_to_csr")},      // C.coo_to_csr
                                         {"to_dense", std::string("coo_to_dense")},  // C.coo_to_dense
                                       }},
                                      {kObjectTypeJTagged, {}},
                                      {kObjectTypeSymbolicKeyType, {}},
                                      {kObjectTypeEnvType, {}}};
  return method_map;
}

BuiltInTypeMap &GetAttrMap() {
  static BuiltInTypeMap attr_map = {
    {kObjectTypeTensorType,
     {
       {"shape", prim::kPrimShape},             // C.shape_
       {"dtype", prim::kPrimDType},             // C.dtype_
       {"size", std::string("size_")},          // C.size_
       {"ndim", std::string("ndim_")},          // C.ndim_
       {"T", std::string("T_")},                // C.T_
       {"itemsize", std::string("itemsize_")},  // C.itemsize_
       {"nbytes", std::string("nbytes_")},      // C.nbytes_
       {"strides", std::string("strides_")},    // C.strides_
     }},
    {kObjectTypeRowTensorType,
     {
       {"values", prim::kPrimRowTensorGetValues},           // F.row_tensor_get_values
       {"indices", prim::kPrimRowTensorGetIndices},         // F.row_tensor_get_indices
       {"dense_shape", prim::kPrimRowTensorGetDenseShape},  // F.row_tensor_get_dense_shape
     }},
    {kObjectTypeCOOTensorType,
     {
       {"values", prim::kPrimCOOTensorGetValues},     // F.coo_tensor_get_values
       {"indices", prim::kPrimCOOTensorGetIndices},   // F.coo_tensor_get_indices
       {"shape", prim::kPrimCOOTensorGetDenseShape},  // F.coo_tensor_get_dense_shape
       {"dtype", std::string("dtype_")},              // C.dtype_
       {"size", std::string("sparse_size_")},         // C.sparse_size_
       {"ndim", std::string("sparse_ndim_")},         // C.sparse_ndim_
       {"itemsize", std::string("itemsize_")},        // C.itemsize_
     }},
    {kObjectTypeCSRTensorType,
     {
       {"indptr", prim::kPrimCSRTensorGetIndptr},     // F.csr_tensor_get_indptr
       {"values", prim::kPrimCSRTensorGetValues},     // F.csr_tensor_get_values
       {"indices", prim::kPrimCSRTensorGetIndices},   // F.csr_tensor_get_indices
       {"shape", prim::kPrimCSRTensorGetDenseShape},  // F.csr_tensor_get_shape
       {"dtype", std::string("dtype_")},              // C.dtype_
       {"size", std::string("sparse_size_")},         // C.sparse_size_
       {"ndim", std::string("sparse_ndim_")},         // C.sparse_ndim_
       {"itemsize", std::string("itemsize_")},        // C.itemsize_
     }},
  };
  return attr_map;
}

Resource::Resource(const py::object &obj)
    : engine_(std::make_shared<abstract::AnalysisEngine>(abstract::GetPrimEvaluatorConstructors(), manager_)),
      source_input_(obj),
      is_cleaned_(false) {}

Resource::~Resource() {
  MS_LOG(DEBUG) << "Resource clear";

  try {
    luojianet_ms::HashMap<std::string, Any>().swap(results_);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Exception when cleaning resource. Error info " << e.what();
  }

  // If exit normally, these global variables will be cleaned
  // in Resource::Clean call by MsPipeline::Compile, but if exit with MS_LOGEXCEPTION,
  // these global variables may not being cleaned, it may
  // cause segmentfault when free python object inside these global variables
  // after python interpreter got freed, so these global variables
  // are cleaned here.
  // So if exit normally, these global variable will be cleaned twice,
  // care be taken to prevent double free in the following functions.
  if (!is_cleaned_) {
    try {
      Clean();
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Exception when cleaning resource. Error info " << e.what();
    } catch (...) {
      MS_LOG(ERROR) << "Exception when cleaning resource.";
    }
  }
}

Any GetMethodOrAttr(const string &name, const TypeId &type_id, const BuiltInTypeMap &method_map) {
  auto type_method_map = method_map.find(static_cast<int64_t>(type_id));
  if (type_method_map == method_map.end()) {
    return Any();
  }
  auto method = type_method_map->second.find(name);
  if (method == type_method_map->second.end()) {
    return Any();
  }
  return method->second;
}

bool Resource::IsTypeInBuiltInMap(const TypeId &type) {
  TypeId type_id = NormalizeTypeId(type);
  const BuiltInTypeMap &method_map = GetMethodMap();
  auto iter = method_map.find(static_cast<int64_t>(type_id));
  if (iter == method_map.end()) {
    const BuiltInTypeMap &attr_map = GetAttrMap();
    iter = attr_map.find(static_cast<int64_t>(type_id));
    if (iter == attr_map.end()) {
      return false;
    }
  }
  return true;
}

Any Resource::GetMethodPtr(const TypeId &type, const std::string &name) {
  TypeId type_id = NormalizeTypeId(type);
  const BuiltInTypeMap &method_map = GetMethodMap();
  return GetMethodOrAttr(name, type_id, method_map);
}

Any Resource::GetAttrPtr(const TypeId &type, const std::string &name) {
  TypeId type_id = NormalizeTypeId(type);
  const BuiltInTypeMap &attr_map = GetAttrMap();
  return GetMethodOrAttr(name, type_id, attr_map);
}

void Resource::GetCompileCacheResource(const py::list &compile_cache_dep_files, const py::dict &weights,
                                       const std::string &queue_name, size_t compile_cache_id,
                                       bool *compile_cache_consistent) {
  compile_cache_manager_ = std::make_shared<CompileCacheManager>(compile_cache_id);
  compile_cache_manager_->InitParallelGroupCkptSaveFile();
  MS_EXCEPTION_IF_NULL(compile_cache_consistent);
  if (!*compile_cache_consistent) {
    MS_LOG(WARNING) << "Check the consistency of dependency files hash failed. Execute all the compilation actions.";
    return;
  }
  compile_cache_manager_->InitCompileCacheHash(compile_cache_dep_files);
  *compile_cache_consistent = compile_cache_manager_->CheckDepFilesHashConsistency();
  if (!*compile_cache_consistent) {
    MS_LOG(WARNING) << "Check the consistency of dependency files hash failed. Execute all the compilation actions.";
    return;
  }
  func_graph_ = compile_cache_manager_->GetCachedFuncGraph(manager_, weights, queue_name);
  layout_map_ = compile_cache_manager_->layout_map();
}

void Resource::CacheFuncGraph() const {
  FuncGraphPtr layout_fg = nullptr;
  std::string parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();
  if (func_graph_->has_flag(parallel::kAutoParallel) &&
      ((parallel_mode == parallel::kAutoParallel) || (parallel_mode == parallel::kSemiAutoParallel))) {
    layout_fg = GetResult(kStepParallelGraph).cast<FuncGraphPtr>();
  }
  compile_cache_manager_->CacheFuncGraph(func_graph_, layout_fg);
}

void Resource::Clean() {
  // AbstractTensor->elements() will be saved in AbstractBasePtrList
  args_spec_.clear();
  source_input_ = py::none();
  // Context with AbstractBasePtrList may be saved in GraphEvaluator
  // some Evaluator like ResolveEvaluator may save Python object in cache,
  // it should be cleaned before Python Interpreter destructed.
  MS_EXCEPTION_IF_NULL(engine_);
  engine_->ClearEvaluatorCache();
  // Clean cache used for parse. As static variable is released after
  // Python threads is released.
  parse::data_converter::ClearObjectCache();
  parse::Parser::CleanParserResource();
  parse::CleanDataClassToClassMap();
  trace::ClearTraceStack();
  is_cleaned_ = true;
}

}  // namespace pipeline
}  // namespace luojianet_ms
