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
#include "frontend/operator/ops_front_infer_function.h"

#include <set>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <map>

#include "abstract/abstract_value.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "abstract/param_validator.h"
#include "pybind_api/ir/tensor_py.h"
#include "frontend/operator/ops.h"
#include "abstract/infer_functions.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace abstract {
enum class State {
  SAME,
  X_ONE,
  Y_ONE,
};

struct SlideInfo {
  int64_t start;
  int64_t step;
  int64_t stop;
};

template <typename T>
AbstractBasePtr InferImplTupleOrListEqual(const std::string &op_name, const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tuples or two lists.
  CheckArgsSize(op_name, args_spec_list, 2);
  auto input_x = CheckArg<T>(op_name, args_spec_list, 0);
  auto input_y = CheckArg<T>(op_name, args_spec_list, 1);
  ValuePtr x_value = input_x->BuildValue();
  ValuePtr y_value = input_y->BuildValue();
  return std::make_shared<AbstractScalar>(*x_value == *y_value);
}

void CheckSlideInput(const ValuePtr &arg_value) {
  MS_EXCEPTION_IF_NULL(arg_value);
  auto value_type = arg_value->type();
  std::string str_type;
  if (value_type) {
    str_type = value_type->ToString();
  } else {
    str_type = "AnyValue";
  }
  MS_LOG(EXCEPTION) << "The type of inputs in range operator only support int64 number. "
                    << "But get a " << str_type << " number.";
}

void CalcSlidePara(const AbstractBasePtrList &args_spec_list, SlideInfo *slide) {
  int64_t arg1 = 0;
  int64_t arg2 = 0;
  if (!args_spec_list.empty()) {
    MS_EXCEPTION_IF_NULL(args_spec_list[0]);
    auto arg_value = args_spec_list[0]->BuildValue();
    if (!arg_value->isa<Int64Imm>()) {
      CheckSlideInput(arg_value);
    }
    arg1 = GetValue<int64_t>(arg_value);
  }

  if (args_spec_list.size() >= 2) {
    MS_EXCEPTION_IF_NULL(args_spec_list[1]);
    auto arg_value = args_spec_list[1]->BuildValue();
    if (!arg_value->isa<Int64Imm>()) {
      CheckSlideInput(arg_value);
    }
    arg2 = GetValue<int64_t>(arg_value);
  }

  if (args_spec_list.size() == 3) {
    MS_EXCEPTION_IF_NULL(args_spec_list[2]);
    auto arg_value = args_spec_list[2]->BuildValue();
    if (!arg_value->isa<Int64Imm>()) {
      CheckSlideInput(arg_value);
    }
    slide->step = GetValue<int64_t>(arg_value);
    slide->start = arg1;
    slide->stop = arg2;
  }

  if (args_spec_list.size() == 2) {
    slide->start = arg1;
    slide->stop = arg2;
  }

  if (args_spec_list.size() == 1) {
    slide->stop = arg1;
  }
}

void ComputeReduceIndex(const std::vector<int64_t> &reverse_x, const std::vector<int64_t> &reverse_y,
                        std::vector<int64_t> *grad_x_reduce_idx, std::vector<int64_t> *grad_y_reduce_idy) {
  MS_EXCEPTION_IF_NULL(grad_x_reduce_idx);
  MS_EXCEPTION_IF_NULL(grad_y_reduce_idy);
  const size_t n = reverse_x.size();
  if (reverse_y.size() < n) {
    MS_LOG(EXCEPTION) << "The size of reverse_y is less than the size of reverse_x.";
  }
  for (size_t i = 0; i < n; ++i) {
    State curr;
    const int64_t x_i = reverse_x[i];
    const int64_t y_i = reverse_y[i];
    const int64_t reduce_idx = SizeToLong(n - 1 - i);
    if (x_i == y_i) {
      curr = State::SAME;
    } else if (x_i == 1) {
      grad_x_reduce_idx->push_back(reduce_idx);
      curr = State::X_ONE;
    } else if (y_i == 1) {
      grad_y_reduce_idy->push_back(reduce_idx);
      curr = State::Y_ONE;
    } else {
      MS_LOG(EXCEPTION) << "not compatible shape input for BroadcastGradientArgs.";
    }
    if (curr == State::SAME && x_i == 1) {
      grad_x_reduce_idx->push_back(reduce_idx);
      grad_y_reduce_idy->push_back(reduce_idx);
      continue;
    }
  }

  std::reverse(grad_x_reduce_idx->begin(), grad_x_reduce_idx->end());
  std::reverse(grad_y_reduce_idy->begin(), grad_y_reduce_idy->end());
}

AbstractBasePtr BroadcastGradientArgsDiff(const std::vector<ValuePtr> &x_shape, const std::vector<ValuePtr> &y_shape) {
  std::vector<int64_t> reverse_x;
  std::vector<int64_t> reverse_y;

  (void)std::transform(x_shape.rbegin(), x_shape.rend(), std::back_inserter(reverse_x),
                       [](const ValuePtr &v) { return v->cast<Int64ImmPtr>()->value(); });
  (void)std::transform(y_shape.rbegin(), y_shape.rend(), std::back_inserter(reverse_y),
                       [](const ValuePtr &v) { return v->cast<Int64ImmPtr>()->value(); });

  if (reverse_x.size() > reverse_y.size()) {
    reverse_y.resize(reverse_x.size(), 1);
  } else {
    reverse_x.resize(reverse_y.size(), 1);
  }

  std::vector<int64_t> grad_x_reduce_idx;
  std::vector<int64_t> grad_y_reduce_idy;
  ComputeReduceIndex(reverse_x, reverse_y, &grad_x_reduce_idx, &grad_y_reduce_idy);

  AbstractBasePtrList abs_list_x;
  AbstractBasePtrList abs_list_y;
  (void)std::transform(grad_x_reduce_idx.begin(), grad_x_reduce_idx.end(), std::back_inserter(abs_list_x),
                       [](int64_t v) { return abstract::FromValue(v); });
  (void)std::transform(grad_y_reduce_idy.begin(), grad_y_reduce_idy.end(), std::back_inserter(abs_list_y),
                       [](int64_t v) { return abstract::FromValue(v); });
  auto x_reduce_idx = std::make_shared<AbstractTuple>(abs_list_x);
  auto y_reduce_idx = std::make_shared<AbstractTuple>(abs_list_y);
  AbstractBasePtrList elem_list;
  elem_list.push_back(x_reduce_idx);
  elem_list.push_back(y_reduce_idx);

  return std::make_shared<AbstractTuple>(elem_list);
}

AbstractBasePtr InferImplTypeof(const AnalysisEnginePtr &, const PrimitivePtr &,
                                const AbstractBasePtrList &args_spec_list) {
  // Inputs: a pointer to an AbstractBase object
  if (args_spec_list.size() != 1) {
    MS_LOG(EXCEPTION) << "The Typeof operator must requires 1 argument, but the size of arguments is "
                      << args_spec_list.size() << ".";
  }
  AbstractBasePtr abs_base = args_spec_list[0];
  MS_EXCEPTION_IF_NULL(abs_base);
  TypePtr type = abs_base->BuildType();
  return std::make_shared<AbstractType>(type);
}

AbstractBasePtr InferImplHasType(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(primitive);
  // Inputs: a pointer to an AbstractBase object and a pointer to a Type
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractTypePtr abs_type = CheckArg<AbstractType>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(abs_type);
  auto mode_v = abs_type->GetValueTrack();
  MS_EXCEPTION_IF_NULL(mode_v);
  if (!mode_v->isa<Type>()) {
    MS_LOG(EXCEPTION) << "Get the type from AbstractType value failed.";
  }

  auto mode_t = mode_v->cast<TypePtr>();
  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  bool v = IsSubtype(args_spec_list[0], mode_t);
  return std::make_shared<AbstractScalar>(std::make_shared<BoolImm>(v), kBool);
}

bool CompareShape(const std::vector<ValuePtr> &x_shape, const std::vector<ValuePtr> &y_shape) {
  if (x_shape.size() != y_shape.size()) {
    return false;
  }

  for (size_t i = 0; i < x_shape.size(); ++i) {
    if (GetValue<int64_t>(x_shape[i]) != GetValue<int64_t>(y_shape[i])) {
      return false;
    }
  }

  return true;
}

AbstractBasePtr DoInferReduceShape(const AbstractTuplePtr &x_shape, const ValuePtr &x_shp_value,
                                   const ValueSequencePtr &axis_value_ptr, const PrimitivePtr &primitive) {
  size_t x_rank = x_shape->size();
  std::set<int64_t> axis_set;
  auto axis_data = axis_value_ptr->value();
  if (axis_data.empty()) {
    int64_t size = 1;
    AbstractBasePtrList values(x_rank, std::make_shared<AbstractScalar>(size));
    return std::make_shared<AbstractTuple>(values);
  }

  for (auto &elem : axis_data) {
    int64_t e_value = CheckAxis(primitive->name(), "axis", elem, -SizeToLong(x_rank), SizeToLong(x_rank), "input_x");
    (void)axis_set.insert(e_value);
  }
  MS_EXCEPTION_IF_NULL(x_shp_value->cast<ValueTuplePtr>());
  auto x_shp_data = x_shp_value->cast<ValueTuplePtr>()->value();
  if (x_shp_data.size() < x_rank) {
    MS_LOG(EXCEPTION) << "x_shape_data.size() " << x_shp_data.size() << " less than x_shape.size() " << x_rank << ".";
  }
  AbstractBasePtrList values;
  for (size_t i = 0; i < x_rank; i++) {
    if (axis_set.count(SizeToLong(i)) || axis_set.count(SizeToLong(i) - SizeToLong(x_rank))) {
      auto axis_v = MakeValue(static_cast<int64_t>(1));
      values.push_back(std::make_shared<AbstractScalar>(axis_v, axis_v->type()));
    } else {
      int64_t dim_value = x_shp_data[i]->cast<Int64ImmPtr>()->value();
      auto dim = MakeValue(dim_value);
      values.push_back(std::make_shared<AbstractScalar>(dim, dim->type()));
    }
  }

  return std::make_shared<AbstractTuple>(values);
}

AbstractBasePtr InferImplBroadcastGradientArgs(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const AbstractBasePtrList &args_spec_list) {
  // this primitive get the index that need to reduce
  // input: x's shape and y's shape, inputs should be tuple
  // output: tuple of x and y 's reduce index, reduce index should be a tuple
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  const size_t inputs_size = 2;
  CheckArgsSize(op_name, args_spec_list, inputs_size);
  auto arg_x = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  auto arg_y = CheckArg<AbstractTuple>(op_name, args_spec_list, 1);

  auto arg_x_value = arg_x->BuildValue()->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(arg_x_value);

  auto arg_y_value = arg_y->BuildValue()->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(arg_y_value);

  const std::vector<ValuePtr> x_shape = arg_x_value->value();
  const std::vector<ValuePtr> y_shape = arg_y_value->value();
  bool is_same_shape = CompareShape(x_shape, y_shape);
  // if it is the same shape , do not need reduce , return empty tuple
  if (is_same_shape) {
    AbstractBasePtrList empty_list;
    auto x_reduce_idx = std::make_shared<AbstractTuple>(empty_list);
    auto y_reduce_idx = std::make_shared<AbstractTuple>(empty_list);

    AbstractBasePtrList elem_list;
    elem_list.push_back(x_reduce_idx);
    elem_list.push_back(y_reduce_idx);

    return std::make_shared<AbstractTuple>(elem_list);
  }
  return BroadcastGradientArgsDiff(x_shape, y_shape);
}

AbstractBasePtr InferImplListMap(const AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list) {
  // Inputs: fn, list1, list2, ...
  MS_EXCEPTION_IF_NULL(engine);
  MS_EXCEPTION_IF_NULL(primitive);
  if (args_spec_list.size() <= 1) {
    MS_LOG(EXCEPTION) << "The ListMap operator requires at least 1 list. But the input size is "
                      << args_spec_list.size() << ".";
  }
  AbstractFunctionPtr fn = CheckArg<AbstractFunction>(primitive->name(), args_spec_list, 0);
  // check args from 1.
  CheckArgsSpec<AbstractList>(AbstractBasePtrList(args_spec_list.begin() + 1, args_spec_list.end()));

  AbstractBasePtrList subargs;
  for (std::size_t i = 1; i < args_spec_list.size(); i++) {
    AbstractListPtr l_ptr = dyn_cast<AbstractList>(args_spec_list[i]);
    if (l_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "The " << i << "th argument of ListMap should be a list, but got "
                        << args_spec_list[i]->ToString() << ".";
    }
    subargs.push_back(AbstractJoin(l_ptr->elements()));
  }
  EvalResultPtr engin_exc = engine->Execute(fn, subargs);
  MS_EXCEPTION_IF_NULL(engin_exc);
  AbstractBasePtrList result;
  for (std::size_t i = 1; i < args_spec_list.size(); i++) {
    result.push_back(engin_exc->abstract());
  }
  return std::make_shared<AbstractList>(result);
}

AbstractBasePtr InferImplListReduce(const AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  // Inputs: a fn, a list and an object of a subclass of a AbstractBase.
  MS_EXCEPTION_IF_NULL(engine);
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  const size_t inputs_size = 3;
  CheckArgsSize(op_name, args_spec_list, inputs_size);
  AbstractFunctionPtr fn = CheckArg<AbstractFunction>(op_name, args_spec_list, 0);
  AbstractListPtr lst = CheckArg<AbstractList>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(lst);
  AbstractBasePtr dflt = args_spec_list[2];

  AbstractBasePtr list_type = AbstractJoin(lst->elements());
  auto result1 = engine->Execute(fn, lst->elements());
  MS_EXCEPTION_IF_NULL(result1);
  auto result2 = engine->Execute(fn, {dflt, list_type});
  MS_EXCEPTION_IF_NULL(result2);
  MS_EXCEPTION_IF_NULL(result1->abstract());
  MS_EXCEPTION_IF_NULL(result2->abstract());
  return result1->abstract()->Join(result2->abstract());
}

AbstractBasePtr InferImplTupleReversed(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractTuplePtr input = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(input);
  auto tuple_elements = input->elements();
  AbstractBasePtrList elem_list;
  (void)std::transform(tuple_elements.rbegin(), tuple_elements.rend(), std::back_inserter(elem_list),
                       [](const AbstractBasePtr &elem) { return elem->Clone(); });
  return std::make_shared<AbstractTuple>(elem_list);
}

AbstractBasePtr InferImplReduceShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  // Inputs: x_shape, axis
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  constexpr size_t arg_size = 2;
  CheckArgsSize(op_name, args_spec_list, arg_size);
  AbstractTuplePtr shape_x = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(shape_x);
  MS_EXCEPTION_IF_NULL(args_spec_list[1]);

  auto x_shp_value = shape_x->BuildValue();
  if (x_shp_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "The ReduceShape operator's data field can't be anything: " << args_spec_list[1]->ToString()
                      << ".";
  }

  // Axis can be scalar, tuple or list
  AbstractSequencePtr axis = nullptr;
  if (args_spec_list[1]->isa<AbstractScalar>()) {
    MS_LOG(DEBUG) << op_name << " evaluator second parameter is scalar.";
    AbstractBasePtrList axis_list = {dyn_cast<AbstractScalar>(args_spec_list[1])};
    axis = std::make_shared<AbstractTuple>(axis_list);
  } else if (args_spec_list[1]->isa<AbstractSequence>()) {
    MS_LOG(DEBUG) << "The type of second argument of ReduceShape operator is sequence.";
    axis = args_spec_list[1]->cast<AbstractSequencePtr>();
  } else {
    MS_LOG(EXCEPTION) << "The second argument of ReduceShape operator should be a scalar or tuple or list, "
                      << "but got " << args_spec_list[1]->ToString() << ".";
  }

  auto axis_value = axis->BuildValue();
  if (axis_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "The ReduceShape operator's data field can't be anything: " << args_spec_list[1]->ToString()
                      << ".";
  }
  auto axis_value_ptr = axis_value->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(axis_value_ptr);
  return DoInferReduceShape(shape_x, x_shp_value, axis_value_ptr, primitive);
}

AbstractBasePtr InferImplTupleDiv(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tuples.
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  constexpr size_t arg_size = 2;
  CheckArgsSize(op_name, args_spec_list, arg_size);
  AbstractTuplePtr shape_x = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  AbstractTuplePtr div_shp = CheckArg<AbstractTuple>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(shape_x);
  MS_EXCEPTION_IF_NULL(div_shp);
  MS_LOG(INFO) << "The shape of dividend:" << shape_x->ToString() << ", the shape of divisor:" << div_shp->ToString();

  auto div_shp_value = div_shp->BuildValue();
  if (div_shp_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "The 'tuple_div' operator shape's data field can't be anything, but got "
                      << args_spec_list[0]->ToString() << ".";
  }

  auto shape_x_value = shape_x->BuildValue();
  if (shape_x_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "The 'tuple_div' operator shape's data field can't be anything, but got "
                      << args_spec_list[1]->ToString() << ".";
  }

  if (div_shp->size() != shape_x->size()) {
    MS_LOG(EXCEPTION) << "The size of inputs of 'tuple_div' operator must be the same, but the size of divisor tuple is"
                      << " " << div_shp->size() << ", the size of dividend tuple is " << shape_x->size() << ".";
  }

  auto shape_x_data = shape_x_value->cast<ValueTuplePtr>()->value();
  auto div_shape_data = div_shp_value->cast<ValueTuplePtr>()->value();
  AbstractBasePtrList values;

  for (size_t i = 0; i < div_shape_data.size(); i++) {
    if (div_shape_data[i]->cast<Int64ImmPtr>() == nullptr) {
      auto value_type = div_shape_data[i]->type();
      std::string str_type;
      if (value_type) {
        str_type = value_type->ToString();
      } else {
        str_type = "AnyValue";
      }
      MS_LOG(EXCEPTION) << "The data type of inputs of 'tuple_div' operator should be an int64 number, but got a "
                        << str_type << " number " << div_shape_data[i]->ToString() << ".";
    }
    int64_t shapex_value = GetValue<int64_t>(shape_x_data[i]);
    int64_t div_value = GetValue<int64_t>(div_shape_data[i]);
    MS_LOG(DEBUG) << "div_shp_shape data shapex_value :" << shapex_value << " div_value: " << div_value;
    if (div_value == 0) {
      MS_LOG(EXCEPTION) << "The divisor value should not be 0!";
    }
    if ((shapex_value % div_value) != 0) {
      MS_LOG(EXCEPTION) << "The inputs of 'tuple_div' operator should be divisible, but they are not divisible now, "
                        << "the dividend is " << shapex_value << ", the divisor is " << div_value << ".";
    }

    int64_t result = shapex_value / div_value;
    auto result_v = MakeValue(result);
    values.push_back(std::make_shared<AbstractScalar>(result_v, result_v->type()));
  }
  return std::make_shared<AbstractTuple>(values);
}

AbstractBasePtr InferImplTuple2Array(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractTuplePtr input = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(input);
  py::tuple data_tuple = ValueToPyData(input->BuildValue());
  py::array data = py::array(data_tuple);
  auto tensor = tensor::TensorPy::MakeTensor(data);
  auto ret = tensor->ToAbstract();
  ret->set_value(tensor);
  MS_LOG(DEBUG) << "The infer result of Tuple2Array operator is tensor, the infer result is " << ret->ToString() << ".";
  return ret;
}

AbstractBasePtr InferImplShapeMul(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple
  // example: tuple = (1, 2, 3), shape_mul(tuple) = 1*2*3 = 6
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractTuplePtr shape_x = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);

  auto shpx_value = shape_x->BuildValue();
  if (shpx_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "The ShapeMul operator shape's data field can't be anything, but got " << shape_x->ToString()
                      << ".";
  }

  auto shpx_data = shpx_value->cast<ValueTuplePtr>()->value();

  int64_t result = 1;
  for (size_t i = 0; i < shpx_data.size(); i++) {
    int64_t value = GetValue<int64_t>(shpx_data[i]);
    result = IntMulWithOverflowCheck(result, value);
  }

  auto result_v = MakeValue(result);
  MS_LOG(DEBUG) << "The infer result of ShapeMul is " << result_v->ToString();
  return std::make_shared<AbstractScalar>(result_v, result_v->type());
}

AbstractBasePtr InferImplSliceGetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  auto op_name = primitive->name();
  constexpr auto slice_getitem_input_size = 2;
  CheckArgsSize(op_name, args_spec_list, slice_getitem_input_size);
  AbstractSlicePtr slice_abs = CheckArg<AbstractSlice>(op_name, args_spec_list, 0);
  const std::map<std::string, AbstractBasePtr> result_map = {
    {kSliceStart, slice_abs->start()}, {kSliceStop, slice_abs->stop()}, {kSliceStep, slice_abs->step()}};
  auto slice_attr = args_spec_list[1]->BuildValue();
  MS_EXCEPTION_IF_NULL(slice_attr);
  if (!slice_attr->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "The second argument of SliceGetItem operator should be a string, but got "
                      << slice_attr->ToString() << ".";
  }
  auto slice_str = GetValue<std::string>(slice_attr);
  auto iter = result_map.find(slice_str);
  if (iter == result_map.end()) {
    MS_EXCEPTION(AttributeError) << "The 'slice' object has no attribute:" << iter->second << ".";
  }
  return iter->second;
}

AbstractBasePtr InferImplMakeSlice(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  // Inputs: three scalars whose value is an int32 number.
  constexpr auto make_slice_input_size = 3;
  CheckArgsSize(primitive->name(), args_spec_list, make_slice_input_size);
  size_t args_size = args_spec_list.size();
  AbstractBasePtrList slice_args;
  for (size_t index = 0; index < args_size; index++) {
    MS_EXCEPTION_IF_NULL(args_spec_list[index]);
    if (args_spec_list[index]->isa<AbstractNone>()) {
      slice_args.push_back(args_spec_list[index]);
    } else if (args_spec_list[index]->isa<AbstractScalar>()) {
      ValuePtr scalar_value = args_spec_list[index]->cast<AbstractScalarPtr>()->BuildValue();
      MS_EXCEPTION_IF_NULL(scalar_value);
      if (scalar_value->isa<IntegerImm>()) {
        slice_args.push_back(args_spec_list[index]);
      } else if (scalar_value->isa<BoolImm>()) {
        ValuePtr scalar_index = MakeValue(static_cast<int64_t>(scalar_value->cast<BoolImmPtr>()->value()));
        slice_args.push_back(scalar_index->ToAbstract());
      } else {
        auto type = scalar_value->type();
        std::string str_type;
        if (type) {
          str_type = type->ToString();
        } else {
          str_type = "AnyValue";
        }
        MS_EXCEPTION(TypeError) << "Slice indices must be integers or bool. But got a " << str_type << " number.";
      }
    } else if (args_spec_list[index]->isa<AbstractTensor>()) {
      auto arg = args_spec_list[index]->cast<AbstractTensorPtr>();
      TypePtr tensor_dtype = arg->element()->BuildType();
      auto build_value = arg->BuildValue();
      MS_EXCEPTION_IF_NULL(build_value);
      auto value = build_value->cast<tensor::TensorPtr>();
      if (value != nullptr) {
        if (value->DataSize() != 1) {
          MS_EXCEPTION(TypeError) << "The input tensor of the MakeSlice operator must contain only one element,"
                                  << "but " << value->ToString() << " has " << value->DataSize() << " elements.";
        }

        if (tensor_dtype->isa<Bool>()) {
          auto *bool_value = static_cast<bool *>(value->data_c());
          slice_args.push_back(MakeValue((static_cast<int64_t>(*bool_value)))->ToAbstract());
        } else if (tensor_dtype == kInt64) {
          auto *int_value = static_cast<int64_t *>(value->data_c());
          slice_args.push_back(MakeValue((*int_value))->ToAbstract());
        } else if (tensor_dtype == kInt32) {
          auto *int_value = static_cast<int32_t *>(value->data_c());
          slice_args.push_back(MakeValue((*int_value))->ToAbstract());
        } else {
          MS_EXCEPTION(TypeError) << "The input tensor type of the MakeSlice operator must be int or bool, but got "
                                  << tensor_dtype->ToString();
        }
      } else {
        slice_args.push_back(args_spec_list[index]);
      }
    } else {
      MS_EXCEPTION(TypeError) << "The " << index << "th input of MakeSlice operator should be scalar, none or tensor, "
                              << "but got " << args_spec_list[index]->ToString() << ".";
    }
  }
  // Slice: start, end, step
  constexpr size_t kMakeSliceInput0 = 0;
  constexpr size_t kMakeSliceInput1 = 1;
  constexpr size_t kMakeSliceInput2 = 2;
  return std::make_shared<AbstractSlice>(slice_args[kMakeSliceInput0], slice_args[kMakeSliceInput1],
                                         slice_args[kMakeSliceInput2]);
}

AbstractBasePtr InferImplMakeRange(const AnalysisEnginePtr &, const PrimitivePtr &,
                                   const AbstractBasePtrList &args_spec_list) {
  if (args_spec_list.empty()) {
    MS_LOG(EXCEPTION) << "For 'range', the arguments could not be empty.";
  }

  constexpr size_t max_args_size = 3;
  if (args_spec_list.size() > max_args_size) {
    MS_LOG(EXCEPTION) << "For 'range', the size of arguments could not exceed 3. But the size of inputs is "
                      << args_spec_list.size() << ".";
  }

  SlideInfo slide = {0, 1, 0};
  CalcSlidePara(args_spec_list, &slide);

  if (slide.step == 0) {
    MS_LOG(EXCEPTION) << "For 'range', the argument 'step' could not be 0.";
  }

  AbstractBasePtrList args;
  if (slide.start <= slide.stop) {
    if (slide.step <= 0) {
      MS_LOG(EXCEPTION) << "For 'range', while the argument 'start' " << slide.start
                        << " is less than or equal to the argument 'stop' " << slide.stop << ", "
                        << "the argument 'step' must be greater than 0, but the argument 'step' is " << slide.step
                        << ".";
    }

    for (int64_t i = slide.start; i < slide.stop; i += slide.step) {
      args.push_back(abstract::FromValue(i));
      if (i > 0 && INT_MAX - i < slide.step) {
        MS_EXCEPTION(ValueError) << "Integer overflow error occurred when traversing the range. "
                                 << "Please check the inputs of range.";
      }
    }
  } else {
    if (slide.step >= 0) {
      MS_LOG(EXCEPTION) << "For 'range', while the argument 'start' " << slide.start << " is greater than the argument "
                        << "'stop' " << slide.stop << ", the argument 'step' must be less than 0, "
                        << "but the argument 'step' is " << slide.step << ".";
    }

    for (int64_t i = slide.start; i > slide.stop; i += slide.step) {
      args.push_back(abstract::FromValue(i));
      if (i < 0 && INT_MIN - i > slide.step) {
        MS_EXCEPTION(ValueError) << "Integer overflow error occurred when traversing the range. "
                                 << "Please check the inputs of range.";
      }
    }
  }

  return std::make_shared<AbstractTuple>(args);
}

AbstractBasePtr InferImplStopGradient(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  // Inputs: any value;
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  return args_spec_list[0]->Clone();
}

AbstractBasePtr InferImplTupleEqual(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  return InferImplTupleOrListEqual<AbstractTuple>(primitive->name(), args_spec_list);
}

AbstractBasePtr InferImplListEqual(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  return InferImplTupleOrListEqual<AbstractList>(primitive->name(), args_spec_list);
}

AbstractBasePtr InferImplStringEqual(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  // Inputs: two scalars whose value is a string.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractScalarPtr scalar_x = CheckArg<AbstractScalar>(op_name, args_spec_list, 0);
  AbstractScalarPtr scalar_y = CheckArg<AbstractScalar>(op_name, args_spec_list, 1);

  ValuePtr value_x = scalar_x->BuildValue();
  ValuePtr value_y = scalar_y->BuildValue();
  if (!value_x->isa<StringImm>() || !value_y->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "The type of two arguments of StringEqual operator requires string, but got param0: "
                      << value_x->ToString() << ", param1: " << value_y->ToString();
  }

  bool ret = (value_x->cast<StringImmPtr>()->value() == value_y->cast<StringImmPtr>()->value());
  return std::make_shared<AbstractScalar>(ret);
}

AbstractBasePtr InferImplStringConcat(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  // Inputs: two scalars whose value is a string.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractScalarPtr scalar_x = CheckArg<AbstractScalar>(op_name, args_spec_list, 0);
  AbstractScalarPtr scalar_y = CheckArg<AbstractScalar>(op_name, args_spec_list, 1);

  ValuePtr value_x = scalar_x->BuildValue();
  ValuePtr value_y = scalar_y->BuildValue();
  if (!value_x->isa<StringImm>() || !value_y->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "The type of two arguments of StringConcat operator requires string, but got param0: "
                      << value_x->ToString() << ", param1: " << value_y->ToString();
  }

  std::string ret = (value_x->cast<StringImmPtr>()->value() + value_y->cast<StringImmPtr>()->value());
  return std::make_shared<AbstractScalar>(ret);
}

AbstractBasePtr InferImplDictLen(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list) {
  return InferTupleOrListOrDictLen<AbstractDictionary>(primitive->name(), args_spec_list);
}

AbstractBasePtr InferImplJ(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const AbstractBasePtrList &args_spec_list) {
  // args: An object of AbstractFunction.
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  MS_LOG(DEBUG) << "evaluate J: " << args_spec_list[0]->ToString();

  AbstractFunctionPtr x = dyn_cast<AbstractFunction>(args_spec_list[0]);
  if (x == nullptr) {
    return std::make_shared<AbstractJTagged>(args_spec_list[0]);
  }

  AbstractFuncAtomPtrList jv;
  auto build_jv = [&jv](const AbstractFuncAtomPtr &func) {
    auto j_closure = std::make_shared<JTransformedAbstractClosure>(func);
    jv.push_back(j_closure);
  };
  x->Visit(build_jv);

  return AbstractFunction::MakeAbstractFunction(jv);
}

AbstractBasePtr InferImplTaylor(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list) {
  // args: An object of AbstractFunction.
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  MS_LOG(DEBUG) << "evaluate Taylor: " << args_spec_list[0]->ToString();

  AbstractFunctionPtr x = dyn_cast<AbstractFunction>(args_spec_list[0]);
  MS_EXCEPTION_IF_NULL(x);

  AbstractFuncAtomPtrList taylor_v;
  auto build_taylor_v = [&taylor_v](const AbstractFuncAtomPtr &func) {
    auto taylor_closure = std::make_shared<TaylorTransformedAbstractClosure>(func);
    taylor_v.push_back(taylor_closure);
  };
  x->Visit(build_taylor_v);

  return AbstractFunction::MakeAbstractFunction(taylor_v);
}

AbstractBasePtr InferImplShard(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const AbstractBasePtrList &args_spec_list) {
  // Inputs: func, in_axes, out_axes, device, level.
  constexpr size_t shard_input_size = 5;
  CheckArgsSize(primitive->name(), args_spec_list, shard_input_size);
  MS_LOG(DEBUG) << "Evaluate Shard: " << args_spec_list[0]->ToString();

  AbstractFunctionPtr x = dyn_cast<AbstractFunction>(args_spec_list[0]);
  MS_EXCEPTION_IF_NULL(x);

  AbstractFuncAtomPtrList shard_v;
  auto build_shard_v = [&shard_v](const AbstractFuncAtomPtr &func) {
    auto shard_closure = std::make_shared<ShardTransformedAbstractClosure>(func);
    shard_v.push_back(shard_closure);
  };
  x->Visit(build_shard_v);

  return AbstractFunction::MakeAbstractFunction(shard_v);
}

AbstractBasePtr InferImplVmap(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list) {
  // args: An object of AbstractFunction.
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  auto fn_arg = args_spec_list[0];
  MS_LOG(DEBUG) << "Evaluate Vmap: " << fn_arg->ToString() << ".";

  AbstractFunctionPtr x = dyn_cast<AbstractFunction>(fn_arg);
  MS_EXCEPTION_IF_NULL(x);

  ValuePtr in_axes = primitive->GetAttr("in_axes");
  ValuePtr out_axes = primitive->GetAttr("out_axes");

  AbstractFuncAtomPtrList vmap_v;
  auto build_vmap_v = [&vmap_v, &in_axes, &out_axes](const AbstractFuncAtomPtr &func) {
    auto vmap_closure = std::make_shared<VmapTransformedAbstractClosure>(func, in_axes, out_axes);
    vmap_v.push_back(vmap_closure);
  };
  x->Visit(build_vmap_v);

  return AbstractFunction::MakeAbstractFunction(vmap_v);
}

AbstractBasePtr InferImplFakeBprop(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor.
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  return args_spec_list[0]->Broaden();
}

// Eval the return type of make_record
AbstractBasePtr InferImplMakeRecord(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  // Inputs: at lease two objects of a subclass of AbstractBase.
  if (args_spec_list.size() < 2) {
    MS_LOG(EXCEPTION) << "The size of arguments of MakeRecord operator must greater than 1, but the input size is "
                      << args_spec_list.size() << ".";
  }

  // args_spec_list[0] maybe AbstractScalarPtr or AbstractTypePtr
  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  TypePtr type = args_spec_list[0]->GetTypeTrack();
  MS_EXCEPTION_IF_NULL(type);
  if (type->type_id() != kMetaTypeTypeType) {
    MS_LOG(EXCEPTION) << "The type of first argument of MakeRecord must be TypeType, but got " << type->ToString();
  }

  auto value_track = args_spec_list[0]->GetValueTrack();
  MS_EXCEPTION_IF_NULL(value_track);
  auto type_ptr = value_track->cast<TypePtr>();
  if (type_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "The value type of first argument of MakeRecord is wrong, the type is "
                      << value_track->ToString();
  }

  auto cls = dyn_cast<Class>(type_ptr);
  MS_EXCEPTION_IF_NULL(cls);
  ClassAttrVector attributes = cls->GetAttributes();
  CheckArgsSize(primitive->name(), args_spec_list, attributes.size() + 1);

  std::vector<AbstractAttribute> abs_attributes;
  for (size_t i = 0; i < attributes.size(); i++) {
    AbstractAttribute elem(attributes[i].first, args_spec_list[i + 1]);
    abs_attributes.push_back(elem);
  }

  return std::make_shared<AbstractClass>(cls->tag(), abs_attributes, cls->methods());
}

REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(TypeOf, prim::kPrimTypeOf, InferImplTypeof, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(HasType, prim::kPrimHasType, InferImplHasType, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(MakeRecord, prim::kPrimMakeRecord, InferImplMakeRecord, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(ListMap, prim::kPrimListMap, InferImplListMap, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(ListReduce, prim::kPrimListReduce, InferImplListReduce, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(TupleReversed, prim::kPrimTupleReversed, InferImplTupleReversed, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(ReducedShape, prim::kPrimReducedShape, InferImplReduceShape, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(TupleDiv, prim::kPrimTupleDiv, InferImplTupleDiv, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(TupleToArray, prim::kPrimTupleToArray, InferImplTuple2Array, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(ShapeMul, prim::kPrimShapeMul, InferImplShapeMul, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(TupleEqual, prim::kPrimTupleEqual, InferImplTupleEqual, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(ListEqual, prim::kPrimListEqual, InferImplListEqual, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(MakeRange, prim::kPrimMakeRange, InferImplMakeRange, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(StopGradient, prim::kPrimStopGradient, InferImplStopGradient, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(StringEqual, prim::kPrimStringEqual, InferImplStringEqual, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(StringConcat, prim::kPrimStringConcat, InferImplStringConcat, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(DictLen, prim::kPrimDictLen, InferImplDictLen, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(FakeBprop, prim::kPrimFakeBprop, InferImplFakeBprop, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(J, prim::kPrimJ, InferImplJ, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(Taylor, prim::kPrimTaylor, InferImplTaylor, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(Shard, prim::kPrimShard, InferImplShard, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(Vmap, prim::kPrimVmap, InferImplVmap, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(BroadcastGradientArgs, prim::kPrimBroadcastGradientArgs,
                                   InferImplBroadcastGradientArgs, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(MakeSlice, prim::kPrimMakeSlice, InferImplMakeSlice, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(SliceGetItem, prim::kPrimSliceGetItem, InferImplSliceGetItem, nullptr);
}  // namespace abstract
}  // namespace mindspore
