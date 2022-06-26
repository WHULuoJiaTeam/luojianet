/**
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

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include "abstract/infer_functions.h"
#include "abstract/utils.h"
#include "abstract/param_validator.h"
#include "utils/shape_utils.h"
#include "ops/op_utils.h"
#include "utils/anf_utils.h"
#include "utils/check_convert_utils.h"

namespace luojianet_ms {
namespace abstract {
namespace {
// Get 3rd argument for UnsortedSegmentOps' inferImpl function
int64_t GetUnsortedSegmentOpScalarArg(const AbstractBasePtrList &args_spec_list, const std::string &op_name) {
  int64_t num_segments_value = 0;
  constexpr size_t scalar_index = 2;
  if (args_spec_list[scalar_index]->isa<AbstractTensor>()) {  // num_segments is Tensor
    auto num_segments = args_spec_list[scalar_index]->cast<AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(num_segments);
    auto num_segments_value_ptr = num_segments->BuildValue();
    MS_EXCEPTION_IF_NULL(num_segments_value_ptr);
    auto num_segments_tensor = num_segments_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(num_segments_tensor);
    if (num_segments->element()->GetTypeTrack()->type_id() == TypeId::kNumberTypeInt64) {
      num_segments_value = *static_cast<int64_t *>(num_segments_tensor->data_c());
    } else {
      num_segments_value = *static_cast<int32_t *>(num_segments_tensor->data_c());
    }
  } else if (args_spec_list[scalar_index]->isa<AbstractScalar>()) {  // num_segments is Scalar
    auto num_segments = CheckArg<AbstractScalar>(op_name, args_spec_list, scalar_index);
    if (num_segments->GetTypeTrack()->type_id() == TypeId::kNumberTypeInt64) {
      num_segments_value = GetValue<int64_t>(num_segments->BuildValue());
    } else {
      num_segments_value = GetValue<int32_t>(num_segments->BuildValue());
    }
  } else {
    MS_LOG(EXCEPTION) << "num_segments incorrect type in " << op_name;
  }
  return num_segments_value;
}
}  // namespace
AbstractBasePtr InferImplScalarToArray(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  // Inputs: a scalar.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractScalarPtr arg = CheckArg<AbstractScalar>(op_name, args_spec_list, 0);
  return std::make_shared<AbstractTensor>(arg, std::make_shared<Shape>());
}

AbstractBasePtr InferImplArrayToScalar(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor with 0 shape.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  auto arg = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto a_shp = arg->shape();
  MS_EXCEPTION_IF_NULL(a_shp);
  if (!a_shp->shape().empty()) {
    MS_LOG(EXCEPTION) << "array_to_scalar requires zero size shape.";
  }
  return arg->element();
}

AbstractBasePtr InferImplBroadCastShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tuples.
  const std::string op_name = primitive->name();
  constexpr size_t args_size = 2;
  CheckArgsSize(op_name, args_spec_list, args_size);
  auto xs = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  auto ys = CheckArg<AbstractTuple>(op_name, args_spec_list, 1);
  auto x_value = xs->BuildValue();
  MS_EXCEPTION_IF_NULL(x_value);
  auto value_tuple_x = x_value->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(value_tuple_x);
  auto shp_tuple_x = value_tuple_x->value();
  ShapeVector shp_x;
  (void)std::transform(std::begin(shp_tuple_x), std::end(shp_tuple_x), std::back_inserter(shp_x),
                       [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });
  auto tupe_value_y = ys->BuildValue();
  MS_EXCEPTION_IF_NULL(tupe_value_y);
  auto value_tuple_y = tupe_value_y->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(value_tuple_y);
  auto shp_tuple_y = value_tuple_y->value();
  ShapeVector shp_y;
  (void)std::transform(std::begin(shp_tuple_y), std::end(shp_tuple_y), std::back_inserter(shp_y),
                       [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });

  ShapeVector res = BroadcastShape(shp_x, shp_y);
  MS_EXCEPTION_IF_NULL(args_spec_list[1]);
  if (res.empty()) {
    MS_LOG(EXCEPTION) << "BroadcastShape fail: " << args_spec_list[0]->ToString() << ","
                      << args_spec_list[1]->ToString();
  }

  AbstractBasePtrList elems;
  (void)std::transform(res.begin(), res.end(), std::back_inserter(elems), [](int64_t n) -> AbstractBasePtr {
    return std::make_shared<AbstractScalar>(std::make_shared<Int64Imm>(n), kInt64);
  });
  return std::make_shared<AbstractTuple>(elems);
}

AbstractBasePtr InferImplStack(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple or tensor.
  const std::string op_name = primitive->name();
  if (args_spec_list.empty()) {
    MS_LOG(EXCEPTION) << "args_spec_list is empty.";
  }

  AbstractTuplePtr arg = nullptr;
  AbstractTensorPtr tensor_base = nullptr;
  size_t tuple_len = 0;
  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  if (args_spec_list[0]->isa<AbstractTuple>()) {
    CheckArgsSize(op_name, args_spec_list, 1);
    arg = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
    tuple_len = arg->elements().size();
    tensor_base = CheckArg<AbstractTensor>(op_name, arg->elements(), 0);
  } else if (args_spec_list[0]->isa<AbstractTensor>()) {
    tuple_len = args_spec_list.size();
    tensor_base = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  }

  MS_EXCEPTION_IF_NULL(tensor_base);
  auto shape = tensor_base->shape();
  MS_EXCEPTION_IF_NULL(shape);
  int64_t rank_base = SizeToLong(shape->shape().size());

  ValuePtr axis = primitive->GetAttr("axis");
  // Axis value should be in [-(rank_base + 1), rank_base).
  int64_t axis_value = CheckAxis(op_name, "axis", axis, -(rank_base + 1), rank_base + 1, "input_x");

  for (size_t i = 1; i < tuple_len; ++i) {
    AbstractTensorPtr tensor = nullptr;
    if (args_spec_list[0]->isa<AbstractTuple>()) {
      tensor = CheckArg<AbstractTensor>(op_name, arg->elements(), i);
    } else if (args_spec_list[0]->isa<AbstractTensor>()) {
      tensor = CheckArg<AbstractTensor>(op_name, args_spec_list, i);
    }
    (void)CheckDtypeSame(op_name, tensor_base, tensor);
    CheckShapeSame(op_name, tensor_base, tensor);
  }
  auto element = tensor_base->element();
  MS_EXCEPTION_IF_NULL(element);
  primitive->set_attr("N", MakeValue(SizeToLong(tuple_len)));
  primitive->set_attr("T", element->BuildType());

  AbstractTensorPtr ret = dyn_cast<AbstractTensor>(tensor_base->Broaden());
  MS_EXCEPTION_IF_NULL(ret);
  auto ret_shape_ptr = ret->shape();
  MS_EXCEPTION_IF_NULL(ret_shape_ptr);
  auto ret_shape = ret_shape_ptr->shape();
  (void)ret_shape.insert(ret_shape.begin() + axis_value, SizeToLong(tuple_len));
  ret->set_shape(std::make_shared<Shape>(ret_shape));
  return ret;
}

AbstractBasePtr InferImplUnique(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list) {
  // inputs: a 1-d Tensor
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractTensorPtr input = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);

  auto shape = input->shape();
  MS_EXCEPTION_IF_NULL(shape);
  if (shape->shape().size() != 1) {
    MS_LOG(EXCEPTION) << "Rank of " << op_name << "'s input must be 1.";
  }
  ShapeVector ids_shape = {Shape::SHP_ANY};
  ShapeVector min_shape = {1};
  ShapeVector max_shape = shape->max_shape();
  if (max_shape.empty()) {
    max_shape = shape->shape();
  }

  auto ids =
    std::make_shared<AbstractTensor>(input->element(), std::make_shared<Shape>(ids_shape, min_shape, max_shape));
  // Currently we choose the same data type as input for the idx.
  TypePtr ids_idx_type = kInt32;
  MS_EXCEPTION_IF_NULL(input->element());
  MS_EXCEPTION_IF_NULL(input->element()->GetTypeTrack());
  if (input->element()->GetTypeTrack()->type_id() == TypeId::kNumberTypeInt64) {
    ids_idx_type = kInt64;
  }
  ShapeVector idx_shape = shape->shape();
  ShapeVector idx_min_shape = shape->min_shape();
  if (idx_min_shape.empty()) {
    idx_min_shape = shape->shape();
  }
  ShapeVector idx_max_shape = shape->max_shape();
  if (idx_max_shape.empty()) {
    idx_max_shape = shape->shape();
  }

  auto ids_idx = std::make_shared<AbstractTensor>(ids_idx_type, idx_shape);
  ids_idx->set_shape(std::make_shared<Shape>(idx_shape, idx_min_shape, idx_max_shape));
  // outputs: ids, ids_idx
  AbstractBasePtrList elements = {ids, ids_idx};
  return std::make_shared<AbstractTuple>(elements);
}

AbstractBasePtr InferImplPadAndShift(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  // inputs: a 1-d Tensor
  const std::string op_name = primitive->name();
  const size_t size_expected = 3;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  AbstractTensorPtr input = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(input);
  auto shape = input->shape();
  MS_EXCEPTION_IF_NULL(shape);
  if (shape->shape().size() != 1) {
    MS_LOG(EXCEPTION) << "Rank of " << op_name << "'s input must be 1.";
  }
  ShapeVector ids_shape = {Shape::SHP_ANY};
  ShapeVector min_shape = {1};
  ShapeVector max_shape = shape->max_shape();
  if (max_shape.empty()) {
    max_shape = shape->shape();
  }
  return std::make_shared<AbstractTensor>(input->element(), std::make_shared<Shape>(ids_shape, min_shape, max_shape));
}

AbstractBasePtr InferImplNonZero(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  const size_t size_expected = 1;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  AbstractTensorPtr x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  auto x_shape = x->shape();
  MS_EXCEPTION_IF_NULL(x_shape);
  ShapeVector y_shape;

  int64_t rank_base = SizeToLong(x_shape->shape().size());
  int64_t max_size = std::accumulate(x_shape->shape().begin(), x_shape->shape().end(), 1, std::multiplies<int64_t>());

  (void)y_shape.emplace_back(rank_base);
  // Indices of elements that are non-zero
  (void)y_shape.emplace_back(Shape::SHP_ANY);

  ShapeVector min_shape = {rank_base, 1};
  ShapeVector max_shape = {rank_base, max_size};

  return std::make_shared<AbstractTensor>(kInt64, std::make_shared<Shape>(y_shape, min_shape, max_shape));
}

AbstractBasePtr InferImplUniqueGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  // inputs: a 1-d Tensor
  const std::string op_name = primitive->name();
  const size_t size_expected = 2;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  AbstractTuplePtr dout = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  CheckArgsSize(op_name + " dout", dout->elements(), size_expected);
  auto ids = CheckArg<AbstractTensor>(op_name, dout->elements(), 0);
  auto ids_idx = CheckArg<AbstractTensor>(op_name, dout->elements(), 1);
  auto ids_shape = ids->shape();
  auto ids_idx_shape = ids_idx->shape();
  MS_EXCEPTION_IF_NULL(ids_shape);
  MS_EXCEPTION_IF_NULL(ids_idx_shape);
  if (ids->shape()->shape().size() != 1) {
    MS_LOG(EXCEPTION) << "Dims of dout[0] of " << op_name << "' input must be 1.";
  }
  if (ids_idx->shape()->shape().size() != 1) {
    MS_LOG(EXCEPTION) << "Dims of dout[1] of " << op_name << "' input must be 1.";
  }

  // outputs: dx
  return std::make_shared<AbstractTensor>(ids->element(), ids_idx->shape());
}

AbstractBasePtr InferImplUnsortedSegmentSum(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  constexpr size_t args_size = 3;
  CheckArgsSize(op_name, args_spec_list, args_size);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  auto segment_ids = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(segment_ids);
  MS_EXCEPTION_IF_NULL(segment_ids->shape());
  auto segment_ids_shape = segment_ids->shape()->shape();
  (void)CheckTensorDType(x, {kFloat16, kFloat32, kFloat64, kInt32}, "Input 0 (x) for UnsortedSegmentSum should be %s");
  (void)CheckTensorDType(segment_ids, {kInt32, kInt64}, "Input 1 (segment_ids) for UnsortedSegmentSum should be %s");
  bool x_is_dyn = (!x->shape()->min_shape().empty() && !x->shape()->max_shape().empty());  // check if dynamic shape
  bool ids_is_dyn = (!segment_ids->shape()->min_shape().empty() && !segment_ids->shape()->max_shape().empty());
  bool op_is_dynamic = x_is_dyn || ids_is_dyn;
  auto x_shape = x->shape()->shape();
  ShapeVector shape;
  int64_t num_segments_value = GetUnsortedSegmentOpScalarArg(args_spec_list, op_name);
  if (num_segments_value <= 0) {
    MS_LOG(EXCEPTION) << "num_segments must be > 0 in UnsortedSegmentSum";
  }
  shape.emplace_back(num_segments_value);
  shape.insert(shape.end(), x_shape.begin() + segment_ids_shape.size(), x_shape.end());
  if (!op_is_dynamic) {  // not dynamic
    for (size_t i = 0; i < segment_ids_shape.size(); i++) {
      if (x_shape[i] != segment_ids_shape[i]) {
        MS_LOG(EXCEPTION) << "Shape values of segments_ids must match with corresponding x shape values";
      }
    }
    return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape));
  }
  ShapeVector min_shape;
  ShapeVector max_shape;
  min_shape.emplace_back(num_segments_value);
  max_shape.emplace_back(num_segments_value);
  bool x_any_shape = std::any_of(x_shape.begin(), x_shape.end(), [](int64_t dim) { return dim == Shape::SHP_ANY; });
  bool ids_any_shape =
    std::any_of(segment_ids_shape.begin(), segment_ids_shape.end(), [](int64_t dim) { return dim == Shape::SHP_ANY; });
  if (!x_any_shape && !ids_any_shape) {  // only validate when shapes fully known
    for (size_t i = 0; i < segment_ids_shape.size(); i++) {
      if (x_shape[i] != segment_ids_shape[i]) {
        MS_LOG(EXCEPTION) << "Shape values of segments_ids must match with corresponding x shape values";
      }
    }
  }
  ShapeVector x_shape_min;
  ShapeVector x_shape_max;
  x_shape_min = (x_is_dyn) ? x->shape()->min_shape() : x->shape()->shape();
  x_shape_max = (x_is_dyn) ? x->shape()->max_shape() : x->shape()->shape();
  min_shape.insert(min_shape.end(), x_shape_min.begin() + segment_ids_shape.size(), x_shape_min.end());
  max_shape.insert(max_shape.end(), x_shape_max.begin() + segment_ids_shape.size(), x_shape_max.end());
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
}

AbstractBasePtr InferImplUnsortedSegmentMax(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  const size_t size_expected = 3;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x->shape());
  auto segment_ids = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(segment_ids);
  MS_EXCEPTION_IF_NULL(segment_ids->shape());
  auto segment_ids_shape = segment_ids->shape()->shape();
  (void)CheckTensorDType(x, {kFloat16, kFloat32, kInt32}, "Input 0 (x) for UnsortedSegmentMax should be %s");
  (void)CheckTensorDType(segment_ids, {kInt32, kInt64}, "Input 1 (segment_ids) for UnsortedSegmentMax should be %s");
  bool x_is_dyn = (!x->shape()->min_shape().empty() && !x->shape()->max_shape().empty());  // check if dynamic
  bool ids_is_dyn = (!segment_ids->shape()->min_shape().empty() && !segment_ids->shape()->max_shape().empty());
  bool op_is_dynamic = x_is_dyn || ids_is_dyn;
  auto x_shape = x->shape()->shape();
  ShapeVector shape;
  int64_t num_segments_value = GetUnsortedSegmentOpScalarArg(args_spec_list, op_name);
  if (num_segments_value <= 0) {
    MS_LOG(EXCEPTION) << "num_segments must be > 0 in UnsortedSegmentMax";
  }
  shape.emplace_back(num_segments_value);
  shape.insert(shape.end(), x_shape.begin() + segment_ids_shape.size(), x_shape.end());
  if (!op_is_dynamic) {  // not dynamic
    if (x_shape[0] != segment_ids_shape[0]) {
      MS_LOG(EXCEPTION) << "Length of segment_ids must match first value of x shape UnsortedSegmentMax";
    }
    return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape));
  }
  ShapeVector min_shape;
  ShapeVector max_shape;
  min_shape.emplace_back(num_segments_value);
  max_shape.emplace_back(num_segments_value);
  bool x_any_shape = std::any_of(x_shape.begin(), x_shape.end(), [](int64_t dim) { return dim == Shape::SHP_ANY; });
  bool ids_any_shape =
    std::any_of(segment_ids_shape.begin(), segment_ids_shape.end(), [](int64_t dim) { return dim == Shape::SHP_ANY; });
  if (!x_any_shape && !ids_any_shape) {
    if (x_shape[0] != segment_ids_shape[0]) {
      MS_LOG(EXCEPTION) << "Length of segment_ids must match first value of x shape UnsortedSegmentMax";
    }
  }
  ShapeVector x_shape_min;
  ShapeVector x_shape_max;
  x_shape_min = (x_is_dyn) ? x->shape()->min_shape() : x->shape()->shape();
  x_shape_max = (x_is_dyn) ? x->shape()->max_shape() : x->shape()->shape();
  min_shape.insert(min_shape.end(), x_shape_min.begin() + segment_ids_shape.size(), x_shape_min.end());
  max_shape.insert(max_shape.end(), x_shape_max.begin() + segment_ids_shape.size(), x_shape_max.end());
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
}

AbstractBasePtr InferImplUnsortedSegmentMin(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  const size_t size_expected = 3;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  auto segment_ids = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(segment_ids);
  MS_EXCEPTION_IF_NULL(segment_ids->shape());
  auto segment_ids_shape = segment_ids->shape()->shape();
  (void)CheckTensorDType(x, {kFloat16, kFloat32, kInt32}, "Input 0 (x) for UnsortedSegmentMin should be %s");
  (void)CheckTensorDType(segment_ids, {kInt32}, "Input 1 (segment_ids) for UnsortedSegmentMin should be %s");
  bool x_is_dyn = (!x->shape()->min_shape().empty() && !x->shape()->max_shape().empty());  // check if dynamic shape
  bool ids_is_dyn = (!segment_ids->shape()->min_shape().empty() && !segment_ids->shape()->max_shape().empty());
  bool op_is_dynamic = x_is_dyn || ids_is_dyn;
  auto x_shape = x->shape()->shape();
  ShapeVector shape;
  int64_t num_segments_value = GetUnsortedSegmentOpScalarArg(args_spec_list, op_name);
  if (num_segments_value <= 0) {
    MS_LOG(EXCEPTION) << "num_segments must be > 0 in UnsortedSegmentMin";
  }
  shape.emplace_back(num_segments_value);
  shape.insert(shape.end(), x_shape.begin() + segment_ids_shape.size(), x_shape.end());
  if (!op_is_dynamic) {  // not dynamic
    if (x_shape[0] != segment_ids_shape[0]) {
      MS_LOG(EXCEPTION) << "Length of segment_ids must match first value of x shape UnsortedSegmentMin";
    }
    return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape));
  }
  ShapeVector min_shape;
  ShapeVector max_shape;
  min_shape.emplace_back(num_segments_value);
  max_shape.emplace_back(num_segments_value);
  bool x_any_shape = std::any_of(x_shape.begin(), x_shape.end(), [](int64_t dim) { return dim == Shape::SHP_ANY; });
  bool ids_any_shape =
    std::any_of(segment_ids_shape.begin(), segment_ids_shape.end(), [](int64_t dim) { return dim == Shape::SHP_ANY; });
  if (!x_any_shape && !ids_any_shape) {  // only validate when shapes fully known
    if (x_shape[0] != segment_ids_shape[0]) {
      MS_LOG(EXCEPTION) << "Length of segment_ids must match first value of x shape UnsortedSegmentMin";
    }
  }
  ShapeVector x_shape_min;
  ShapeVector x_shape_max;
  x_shape_min = (x_is_dyn) ? x->shape()->min_shape() : x->shape()->shape();
  x_shape_max = (x_is_dyn) ? x->shape()->max_shape() : x->shape()->shape();
  min_shape.insert(min_shape.end(), x_shape_min.begin() + segment_ids_shape.size(), x_shape_min.end());
  max_shape.insert(max_shape.end(), x_shape_max.begin() + segment_ids_shape.size(), x_shape_max.end());
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
}

AbstractBasePtr InferImplScatterAdd(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  constexpr auto kScatterAddInputNum = 3;
  const std::string op_name = primitive->name();
  CheckRequiredArgsSize(op_name, args_spec_list, kScatterAddInputNum);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  ShapeVector shape = x->shape()->shape();
  ShapeVector min_shape = x->shape()->min_shape();
  ShapeVector max_shape = x->shape()->max_shape();
  CheckMinMaxShape(shape, &min_shape, &max_shape);
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
}

AbstractBasePtr InferImplScatterSub(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  constexpr auto kScatterSubInputNum = 3;
  const std::string op_name = primitive->name();
  CheckRequiredArgsSize(op_name, args_spec_list, kScatterSubInputNum);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  ShapeVector shape = x->shape()->shape();
  ShapeVector min_shape = x->shape()->min_shape();
  ShapeVector max_shape = x->shape()->max_shape();
  CheckMinMaxShape(shape, &min_shape, &max_shape);
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
}

AbstractBasePtr InferImplScatterUpdate(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckRequiredArgsSize(op_name, args_spec_list, 3);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  ShapeVector shape = x->shape()->shape();
  ShapeVector min_shape = x->shape()->min_shape();
  ShapeVector max_shape = x->shape()->max_shape();
  CheckMinMaxShape(shape, &min_shape, &max_shape);
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
}

AbstractBasePtr InferImplScatterElements(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckRequiredArgsSize(op_name, args_spec_list, kSizeThree);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, kIndexZero);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  ShapeVector shape = x->shape()->shape();
  ShapeVector min_shape = x->shape()->min_shape();
  ShapeVector max_shape = x->shape()->max_shape();
  CheckMinMaxShape(shape, &min_shape, &max_shape);
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
}

AbstractBasePtr InferImplMapCacheIdx(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  const size_t size_expected = 5;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  auto hash_map = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(hash_map->shape());

  auto indices = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  auto indices_shp = indices->shape();
  MS_EXCEPTION_IF_NULL(indices_shp);

  ShapeVector shape;
  ShapeVector min_shape;
  ShapeVector max_shape;
  if (!indices_shp->max_shape().empty()) {
    max_shape = indices_shp->max_shape();
  } else {
    max_shape = indices_shp->shape();
  }
  for (size_t i = 0; i < max_shape.size(); i++) {
    shape.emplace_back(Shape::SHP_ANY);
    min_shape.emplace_back(1);
  }

  auto cache_idx = std::make_shared<AbstractTensor>(hash_map->element(), indices->shape());
  auto old_emb_idx =
    std::make_shared<AbstractTensor>(hash_map->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
  auto miss_emb_idx =
    std::make_shared<AbstractTensor>(hash_map->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
  auto swap_emb_idx =
    std::make_shared<AbstractTensor>(hash_map->element(), std::make_shared<Shape>(shape, min_shape, max_shape));

  AbstractBasePtrList elements = {cache_idx, old_emb_idx, miss_emb_idx, swap_emb_idx};
  return std::make_shared<AbstractTuple>(elements);
}

AbstractBasePtr InferImplCacheSwapTable(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  const size_t size_expected = 3;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  auto cache_table = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto cache_table_shp = cache_table->shape();
  MS_EXCEPTION_IF_NULL(cache_table_shp);

  auto swap_cache_idx = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  auto swap_cache_idx_shp = swap_cache_idx->shape();
  MS_EXCEPTION_IF_NULL(swap_cache_idx_shp);

  auto cache_table_shape = cache_table_shp->shape();
  auto swap_cache_idx_shape = swap_cache_idx_shp->shape();
  ShapeVector shape;
  shape.emplace_back(swap_cache_idx_shape[0]);
  shape.emplace_back(cache_table_shape[1]);
  auto swap_cache_idx_max_shape = swap_cache_idx_shp->max_shape();
  ShapeVector max_shape;
  ShapeVector min_shape;
  if (!swap_cache_idx_max_shape.empty()) {
    max_shape.emplace_back(swap_cache_idx_max_shape[0]);
    max_shape.emplace_back(cache_table_shape[1]);
  } else {
    max_shape = shape;
  }
  for (size_t i = 0; i < max_shape.size(); ++i) {
    min_shape.emplace_back(1);
  }

  AbstractTensorPtr ret =
    std::make_shared<AbstractTensor>(cache_table->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
  return ret;
}

AbstractBasePtr InferImplUpdateCache(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  auto input_x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);

  ShapeVector shape;
  shape.emplace_back(1);

  AbstractTensorPtr ret = std::make_shared<AbstractTensor>(input_x->element(), std::make_shared<Shape>(shape));
  return ret;
}

AbstractBasePtr InferImplSubAndFilter(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  auto input_x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto input_x_shp = input_x->shape();
  MS_EXCEPTION_IF_NULL(input_x_shp);

  ShapeVector shape;
  ShapeVector min_shape;
  ShapeVector max_shape;
  if (!input_x_shp->max_shape().empty()) {
    max_shape = input_x_shp->max_shape();
  } else {
    max_shape = input_x_shp->shape();
  }
  for (size_t i = 0; i < max_shape.size(); i++) {
    shape.emplace_back(Shape::SHP_ANY);
    min_shape.emplace_back(1);
  }
  auto filter_res =
    std::make_shared<AbstractTensor>(input_x->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
  auto filter_idx =
    std::make_shared<AbstractTensor>(input_x->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
  AbstractBasePtrList elements = {filter_res, filter_idx};
  return std::make_shared<AbstractTuple>(elements);
}

AbstractBasePtr InferImplDiv(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  const size_t size_expected = 2;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto y = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  MS_EXCEPTION_IF_NULL(y);
  MS_EXCEPTION_IF_NULL(y->shape());
  ShapeVector x_shape = x->shape()->shape();
  ShapeVector y_shape = y->shape()->shape();
  ShapeVector out_shape = BroadcastShape(x_shape, y_shape);
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(out_shape));
}

AbstractBasePtr InferImplRealDiv(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  const size_t size_expected = 2;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto y = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  MS_EXCEPTION_IF_NULL(y);
  MS_EXCEPTION_IF_NULL(y->shape());
  ShapeVector x_shape = x->shape()->shape();
  ShapeVector y_shape = y->shape()->shape();
  ShapeVector out_shape = BroadcastShape(x_shape, y_shape);
  if (out_shape.empty()) {
    MS_LOG(EXCEPTION) << "BroadcastShape fail: " << args_spec_list[0]->ToString() << ","
                      << args_spec_list[1]->ToString();
  }
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(out_shape));
}

AbstractBasePtr InferImplGatherV2(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  const std::string &op_name = primitive->name();
  constexpr size_t args_size = 3;
  CheckArgsSize(op_name, args_spec_list, args_size);
  AbstractTensorPtr params = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  AbstractTensorPtr indices = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  bool ind_dyn = (!indices->shape()->min_shape().empty() && !indices->shape()->max_shape().empty());
  bool param_dyn = (!params->shape()->min_shape().empty() && !params->shape()->max_shape().empty());
  int64_t axis_val = 0;
  // 3rd input is a Tensor when GatherV2 is a dynamic shape operator
  constexpr size_t aixs_index = 2;
  if (args_spec_list[aixs_index]->isa<AbstractTensor>()) {
    auto axis = args_spec_list[aixs_index]->cast<AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(axis);
    auto axis_value_ptr = axis->BuildValue();
    MS_EXCEPTION_IF_NULL(axis_value_ptr);
    auto axis_tensor = axis_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(axis_tensor);
    axis_val = *static_cast<int64_t *>(axis_tensor->data_c());
  } else if (args_spec_list[aixs_index]->isa<AbstractScalar>()) {
    auto axis = args_spec_list[aixs_index]->cast<AbstractScalarPtr>();
    axis_val = GetValue<int64_t>(axis->BuildValue());
  } else {
    MS_LOG(EXCEPTION) << "Invalid abstract type:" << args_spec_list[2]->type_name();
  }
  auto params_shp = params->shape()->shape();
  auto indices_shp = indices->shape()->shape();
  auto params_rank = static_cast<int64_t>(params_shp.size());
  // either inputs or both can be dynamic and computation requires min/max shapes for both
  ShapeVector param_shp_min = (param_dyn) ? params->shape()->min_shape() : params->shape()->shape();
  ShapeVector param_shp_max = (param_dyn) ? params->shape()->max_shape() : params->shape()->shape();
  ShapeVector indices_shp_min = (ind_dyn) ? indices->shape()->min_shape() : indices->shape()->shape();
  ShapeVector indices_shp_max = (ind_dyn) ? indices->shape()->max_shape() : indices->shape()->shape();
  // check axis_val within interval: [-params_rank, params_rank)
  if (-params_rank > axis_val || axis_val >= params_rank) {
    MS_LOG(EXCEPTION) << "For Gather - Axis value must be within [ " << -params_rank << ", " << params_rank << " ) "
                      << "Got " << axis_val << ".";
  }
  if (axis_val < 0) {
    axis_val += params_rank;
  }
  auto calc_shape = [axis_val](const ShapeVector &ind_vec, const ShapeVector &params_vec) -> ShapeVector {
    ShapeVector out_vec;
    std::copy(params_vec.begin(), params_vec.begin() + axis_val, std::back_inserter(out_vec));
    copy(ind_vec.begin(), ind_vec.end(), std::back_inserter(out_vec));
    copy(params_vec.begin() + axis_val + 1, params_vec.end(), std::back_inserter(out_vec));
    return out_vec;
  };
  ShapeVector out_shape = calc_shape(indices_shp, params_shp);
  if (ind_dyn || param_dyn) {
    ShapeVector min_shape = calc_shape(indices_shp_min, param_shp_min);
    ShapeVector max_shape = calc_shape(indices_shp_max, param_shp_max);
    return std::make_shared<AbstractTensor>(params->element(),
                                            std::make_shared<Shape>(out_shape, min_shape, max_shape));
  }
  return std::make_shared<AbstractTensor>(params->element(), std::make_shared<Shape>(out_shape));
}

AbstractBasePtr InferImplDynamicAssign(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor
  const size_t size_expected = 2;
  CheckArgsSize(primitive->name(), args_spec_list, size_expected);

  MS_LOG(INFO) << "InferImplDynamicAssign " << args_spec_list[0];
  auto type = args_spec_list[0]->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  if (type->type_id() == kObjectTypeRefKey) {
    return args_spec_list[1]->Broaden();
  } else {
    auto x = CheckArg<AbstractTensor>(primitive->name(), args_spec_list, 0);
    auto y = CheckArg<AbstractTensor>(primitive->name(), args_spec_list, 1);
    MS_EXCEPTION_IF_NULL(x);
    MS_EXCEPTION_IF_NULL(y);
    auto y_shape = y->shape();
    MS_EXCEPTION_IF_NULL(y_shape);
    if (!y_shape->max_shape().empty()) {
      x->set_shape(y->shape());
    }
    return args_spec_list[0];
  }
}

AbstractBasePtr InferImplEmbeddingLookup(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  auto params = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto params_shp = params->shape();
  MS_EXCEPTION_IF_NULL(params_shp);
  auto params_shape = params_shp->shape();
  auto indices = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  auto indices_shp = indices->shape();
  MS_EXCEPTION_IF_NULL(indices_shp);
  auto indices_shape = indices_shp->shape();
  auto indices_max_shape = indices_shp->max_shape();
  auto indices_min_shape = indices_shp->min_shape();
  ShapeVector shape;
  ShapeVector max_shape;
  ShapeVector min_shape;
  shape.insert(shape.end(), indices_shape.begin(), indices_shape.end());
  shape.insert(shape.end(), params_shape.begin() + 1, params_shape.end());
  if (!indices_max_shape.empty()) {
    max_shape.insert(max_shape.end(), indices_max_shape.begin(), indices_max_shape.end());
    max_shape.insert(max_shape.end(), params_shape.begin() + 1, params_shape.end());
  } else {
    max_shape = shape;
  }
  if (!indices_min_shape.empty()) {
    min_shape.insert(min_shape.end(), indices_min_shape.begin(), indices_min_shape.end());
    min_shape.insert(min_shape.end(), params_shape.begin() + 1, params_shape.end());
  } else {
    min_shape = shape;
  }

  AbstractTensorPtr ret =
    std::make_shared<AbstractTensor>(params->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
  return ret;
}

AbstractBasePtr InferImplTranspose(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  const std::string &op_name = primitive->name();
  AbstractTensorPtr input = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto input_shp = input->shape()->shape();
  ValuePtr perm = primitive->GetAttr("perm");
  MS_EXCEPTION_IF_NULL(perm);
  auto perm_val = perm->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(perm_val);
  auto perm_val_data = perm_val->value();
  ShapeVector perm_vec;
  (void)std::transform(std::begin(perm_val_data), std::end(perm_val_data), std::back_inserter(perm_vec),
                       [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });
  ShapeVector result_shp;
  ShapeVector max_shp;
  ShapeVector min_shp;
  ShapeVector x_max_shp = input->shape()->max_shape();
  ShapeVector x_min_shp = input->shape()->min_shape();
  CheckMinMaxShape(input_shp, &x_min_shp, &x_max_shp);
  for (size_t i = 0; i < perm_vec.size(); i++) {
    auto idx = static_cast<size_t>(perm_vec[i]);
    result_shp.push_back(input_shp[idx]);
    max_shp.push_back(x_max_shp[idx]);
    min_shp.push_back(x_min_shp[idx]);
  }
  return std::make_shared<AbstractTensor>(input->element(), std::make_shared<Shape>(result_shp, min_shp, max_shp));
}

static ShapeVector GetShape(const PrimitivePtr &primitive, const AbstractBasePtrList &args_spec_list,
                            const std::string &op_name) {
  ShapeVector shape;
  if (args_spec_list.size() == kSizeTwo) {
    auto input_value = args_spec_list[1]->BuildValue();
    if (input_value->isa<tensor::Tensor>()) {
      shape = CheckAndConvertUtils::CheckTensorIntValue("shape", input_value, op_name);
    } else {
      shape = CheckAndConvertUtils::CheckTupleInt("input[shape]", input_value, op_name);
    }
  } else {
    ValuePtr sh = primitive->GetAttr("shape");
    MS_EXCEPTION_IF_NULL(sh);
    if (sh->isa<ValueTuple>()) {
      auto reshape_value_tuple = sh->cast<ValueTuplePtr>();
      MS_EXCEPTION_IF_NULL(reshape_value_tuple);
      auto reshape_tuple = reshape_value_tuple->value();
      (void)std::transform(std::begin(reshape_tuple), std::end(reshape_tuple), std::back_inserter(shape),
                           [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });
    } else if (sh->isa<tensor::Tensor>()) {
      shape = CheckAndConvertUtils::CheckTensorIntValue("shape", sh, "Reshape");
    } else {
      MS_EXCEPTION(ValueError) << "In stage of execution， the primitive[Reshape]'s input['shape'] must be a tuple or "
                               << "constant Tensor.";
    }
  }
  return shape;
}

AbstractBasePtr InferImplReshape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());

  ShapeVector shape = GetShape(primitive, args_spec_list, op_name);
  ShapeVector x_shape = x->shape()->shape();
  ShapeVector x_max_shape = x->shape()->max_shape();
  ShapeVector x_min_shape = x->shape()->min_shape();
  if (x_max_shape.empty()) {
    x_max_shape = x_shape;
  }
  if (x_min_shape.empty()) {
    x_min_shape = x_shape;
  }

  auto max_shape = shape;
  auto min_shape = shape;
  int64_t x_num = 1;
  int64_t x_min_num = 1;
  int64_t x_max_num = 1;
  for (int64_t value : x_shape) {
    x_num = LongMulWithOverflowCheck(value, x_num);
  }
  for (int64_t value : x_min_shape) {
    x_min_num = LongMulWithOverflowCheck(value, x_min_num);
  }
  for (int64_t value : x_max_shape) {
    x_max_num = LongMulWithOverflowCheck(value, x_max_num);
  }

  auto it_first = find(shape.begin(), shape.end(), -1);
  if (it_first != shape.end()) {
    auto it_second = find(it_first + 1, shape.end(), -1);
    if (it_second != shape.end()) {
      MS_LOG(EXCEPTION) << "At most one component of input shape can be -1, but got " << shape;
    }
    auto index = LongToSize(std::distance(shape.begin(), it_first));
    int64_t infer_value = x_num;
    int64_t infer_min_value = x_min_num;
    int64_t infer_max_value = x_max_num;
    for (size_t i = 0; i < shape.size(); ++i) {
      int64_t value = shape[i];
      if (value != -1 && value != 0) {
        infer_value = infer_value / value;
        infer_min_value = infer_min_value / value;
        infer_max_value = infer_max_value / value;
      }
    }
    shape[index] = infer_value;
    min_shape[index] = infer_min_value;
    max_shape[index] = infer_max_value;
  }

  int64_t shape_num = 1;
  for (int64_t value : shape) {
    shape_num = LongMulWithOverflowCheck(value, shape_num);
  }
  if (shape_num != x_num) {
    MS_LOG(EXCEPTION) << "The accumulate of x_shape must be equal to out_shape, but got x_shape: " << x_shape
                      << ", and out_shape: " << shape;
  }

  AbstractTensorPtr ret =
    std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
  return ret;
}

AbstractBasePtr InferImplMapUniform(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  // Inputs: one tensor.
  const std::string op_name = primitive->name();
  const size_t size_expected = 3;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  return args_spec_list[0]->Broaden();
}

AbstractBasePtr InferImplSplit(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractTensorPtr input_x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  ShapeVector x_shape = input_x->shape()->shape();
  ShapeVector x_shape_min = input_x->shape()->min_shape();
  if (x_shape_min.empty()) {
    x_shape_min = x_shape;
  }
  ShapeVector x_shape_max = input_x->shape()->max_shape();
  if (x_shape_max.empty()) {
    x_shape_max = x_shape;
  }
  int64_t rank = SizeToLong(x_shape.size());

  ValuePtr axis = primitive->GetAttr("axis");
  int64_t axis_value_pos = CheckAxis(op_name, "axis", axis, -(rank + 1), rank, "input_x");
  int64_t output_num_value = GetValue<int64_t>(primitive->GetAttr("output_num"));
  size_t pos = LongToSize(axis_value_pos);
  if ((x_shape[pos] != Shape::SHP_ANY) && (x_shape[pos] % output_num_value != 0)) {
    MS_LOG(EXCEPTION) << "x_shape[" << pos << "] = " << x_shape[pos]
                      << " must be divisible by output_num = " << output_num_value;
  }

  ShapeVector output_shape = x_shape;
  if (output_shape[pos] != Shape::SHP_ANY) {
    output_shape[pos] = static_cast<int>(x_shape[pos] / output_num_value);
  }
  ShapeVector output_shape_min = x_shape_min;
  output_shape_min[pos] = static_cast<int>(x_shape_min[pos] / output_num_value);
  ShapeVector output_shape_max = x_shape_max;
  output_shape_max[pos] = static_cast<int>(x_shape_max[pos] / output_num_value);

  AbstractBasePtrList output_list;
  for (int64_t i = 0; i < output_num_value; ++i) {
    auto output = input_x->Broaden();
    output->set_shape(std::make_shared<Shape>(output_shape, output_shape_min, output_shape_max));
    output_list.push_back(output);
  }
  return std::make_shared<AbstractTuple>(output_list);
}

AbstractBasePtr InferImplSequenceMask(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  const std::string &op_name = primitive->name();
  const size_t size_expected = 2;
  CheckArgsSize(op_name, args_spec_list, size_expected);

  AbstractTensorPtr lengths = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  (void)CheckTensorDType(lengths, {kInt32, kInt64}, "Input 1 (lengths) for SequenceMask should be one of: %s");

  int64_t maxlen_value = 0;

  if (args_spec_list[1]->isa<AbstractScalar>()) {
    AbstractScalarPtr maxlen = CheckArg<AbstractScalar>(op_name, args_spec_list, 1);
    (void)CheckScalarType(maxlen, {kInt32, kInt64}, "Input 0 (maxlen) for SequenceMask should be one of: %s");

    TypePtr maxlen_type = nullptr;
    maxlen_type = maxlen->GetTypeTrack();
    MS_EXCEPTION_IF_NULL(maxlen_type);

    if (maxlen_type->type_id() == TypeId::kNumberTypeInt32) {
      maxlen_value = static_cast<int64_t>(GetValue<int32_t>(maxlen->BuildValue()));
    } else if (maxlen_type->type_id() == TypeId::kNumberTypeInt64) {
      maxlen_value = GetValue<int64_t>(maxlen->BuildValue());
    }
  } else if (args_spec_list[1]->isa<AbstractTensor>()) {
    auto maxlen_tensor_ptr = args_spec_list[1]->cast<AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(maxlen_tensor_ptr);
    auto maxlen_value_ptr = maxlen_tensor_ptr->BuildValue();
    MS_EXCEPTION_IF_NULL(maxlen_value_ptr);
    auto maxlen_tensor = maxlen_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(maxlen_tensor);
    maxlen_value = *static_cast<int64_t *>(maxlen_tensor->data_c());
  }

  if (maxlen_value <= 0) {
    MS_LOG(EXCEPTION) << "maxlen must be positive, but got: " << maxlen_value;
  }

  ShapeVector lengths_shape = lengths->shape()->shape();
  ShapeVector lengths_shape_min = lengths->shape()->min_shape();
  if (lengths_shape_min.empty()) {
    lengths_shape_min = lengths_shape;
  }
  ShapeVector lengths_shape_max = lengths->shape()->max_shape();
  if (lengths_shape_max.empty()) {
    lengths_shape_max = lengths_shape;
  }

  lengths_shape.push_back(maxlen_value);
  lengths_shape_min.push_back(maxlen_value);
  lengths_shape_max.push_back(maxlen_value);

  ShapePtr output_shape = std::make_shared<Shape>(lengths_shape, lengths_shape_min, lengths_shape_max);
  return std::make_shared<AbstractTensor>(kBool, output_shape);
}

AbstractBasePtr InferImplConcatOffset(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  if (args_spec_list.empty()) {
    MS_LOG(EXCEPTION) << "args_spec_list is empty.";
  }

  AbstractTuplePtr arg = nullptr;
  AbstractTensorPtr tensor_base = nullptr;
  size_t tuple_len = 0;
  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  if (args_spec_list[0]->isa<AbstractTuple>()) {
    CheckArgsSize(op_name, args_spec_list, 1);
    arg = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
    tuple_len = arg->elements().size();
    tensor_base = CheckArg<AbstractTensor>(op_name, arg->elements(), 0);
  } else if (args_spec_list[0]->isa<AbstractTensor>()) {
    tuple_len = args_spec_list.size();
    tensor_base = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  }

  MS_EXCEPTION_IF_NULL(tensor_base);
  ShapeVector shape_base = tensor_base->shape()->shape();
  size_t rank = shape_base.size();
  ShapeVector out_shape{SizeToLong(tuple_len), SizeToLong(rank)};
  TypePtr out_type = kInt64;
  return std::make_shared<AbstractTensor>(out_type, std::make_shared<Shape>(out_shape));
}

AbstractBasePtr InferImplConcat(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  if (args_spec_list.empty()) {
    MS_LOG(EXCEPTION) << "args_spec_list is empty.";
  }

  AbstractTuplePtr arg = nullptr;
  AbstractTensorPtr tensor_base = nullptr;
  size_t tuple_len = 0;
  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  if (args_spec_list[0]->isa<AbstractTuple>()) {
    CheckArgsSize(op_name, args_spec_list, 1);
    arg = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
    tuple_len = arg->elements().size();
    tensor_base = CheckArg<AbstractTensor>(op_name, arg->elements(), 0);
  } else if (args_spec_list[0]->isa<AbstractTensor>()) {
    tuple_len = args_spec_list.size();
    tensor_base = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  }

  MS_EXCEPTION_IF_NULL(tensor_base);
  ShapeVector shape_base = tensor_base->shape()->shape();
  int64_t rank_base = SizeToLong(shape_base.size());
  ShapeVector min_shape_base = tensor_base->shape()->min_shape();
  ShapeVector max_shape_base = tensor_base->shape()->max_shape();
  CheckMinMaxShape(shape_base, &min_shape_base, &max_shape_base);

  primitive->set_attr("T", tensor_base->element()->BuildType());
  primitive->set_attr("inputNums", MakeValue(SizeToLong(tuple_len)));

  ValuePtr axis = primitive->GetAttr("axis");
  // Axis value should be in [-(rank_base + 1), rank_base).
  int64_t axis_value = CheckAxis(op_name, "axis", axis, -(rank_base + 1), rank_base, "input_x");

  int64_t all_shp = shape_base[axis_value];
  int64_t min_all_shp = min_shape_base[axis_value];
  int64_t max_all_shp = max_shape_base[axis_value];
  for (size_t i = 1; i < tuple_len; ++i) {
    AbstractTensorPtr tensor = nullptr;
    if (args_spec_list[0]->isa<AbstractTuple>()) {
      tensor = CheckArg<AbstractTensor>(op_name, arg->elements(), i);
    } else if (args_spec_list[0]->isa<AbstractTensor>()) {
      tensor = CheckArg<AbstractTensor>(op_name, args_spec_list, i);
    }
    ShapeVector shape_tensor = tensor->shape()->shape();
    int64_t rank_tensor = SizeToLong(shape_tensor.size());
    ShapeVector min_shape_tensor = tensor->shape()->min_shape();
    ShapeVector max_shape_tensor = tensor->shape()->max_shape();
    CheckMinMaxShape(shape_tensor, &min_shape_tensor, &max_shape_tensor);
    (void)CheckDtypeSame(op_name, tensor_base, tensor);
    if (rank_tensor != rank_base) {
      MS_LOG(EXCEPTION) << op_name << " can not concat element " << i << " with the first element: Wrong Rank";
    }
    for (int j = 0; j < rank_base; ++j) {
      if (j != axis_value && shape_tensor[j] != shape_base[j]) {
        MS_LOG(EXCEPTION) << op_name << " can not concat element " << i << " with the first element: Wrong Size";
      }
    }
    if (all_shp == -1 || shape_base[axis_value] == -1) {
      all_shp = -1;
    } else {
      all_shp += shape_tensor[axis_value];
    }
    min_all_shp += min_shape_tensor[axis_value];
    max_all_shp += max_shape_tensor[axis_value];
  }

  AbstractTensorPtr ret = dyn_cast<AbstractTensor>(tensor_base->Broaden());
  MS_EXCEPTION_IF_NULL(ret);
  auto shape = ret->shape()->shape();
  auto min_shape = ret->shape()->min_shape();
  auto max_shape = ret->shape()->max_shape();
  CheckMinMaxShape(shape, &min_shape, &max_shape);
  shape[axis_value] = all_shp;
  min_shape[axis_value] = min_all_shp;
  max_shape[axis_value] = max_all_shp;
  ret->set_shape(std::make_shared<Shape>(shape, min_shape, max_shape));
  return ret;
}

AbstractBasePtr InferImplRange(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const AbstractBasePtrList &args_spec_list) {
  const std::string &op_name = primitive->name();
  if (args_spec_list.size() == 1) {
    return args_spec_list[0]->Broaden();
  }
  constexpr size_t args_size = 3;
  constexpr size_t range_start_index = 0;
  constexpr size_t range_end_index = 1;
  constexpr size_t range_delta_index = 2;
  CheckArgsSize(op_name, args_spec_list, args_size);
  AbstractTensorPtr range_start = CheckArg<AbstractTensor>(op_name, args_spec_list, range_start_index);
  AbstractTensorPtr range_end = CheckArg<AbstractTensor>(op_name, args_spec_list, range_end_index);
  AbstractTensorPtr range_delta = CheckArg<AbstractTensor>(op_name, args_spec_list, range_delta_index);

  TypePtrList supported_types = {kInt64, kInt32, kFloat32, kFloat64};
  TypePtr range_start_type = CheckTensorDType(range_start, supported_types, "range_start input of Range should be %s");
  TypePtr range_end_type = CheckTensorDType(range_end, supported_types, "range_start input of Range should be %s");
  TypePtr range_delta_type = CheckTensorDType(range_delta, supported_types, "range_start input of Range should be %s");
  // check all 3 inputs are same type
  if (!IsIdentidityOrSubclass(range_start_type, range_end_type) ||
      !IsIdentidityOrSubclass(range_end_type, range_delta_type)) {
    MS_LOG(EXCEPTION) << "All inputs must have same type, but got: " << args_spec_list[range_start_index]->type_name()
                      << ", " << args_spec_list[range_end_index]->type_name() << ", and "
                      << args_spec_list[range_delta_index]->type_name();
  }

  ValuePtr max_output_length_ptr = primitive->GetAttr("maxlen");
  int64_t max_output_length = GetValue<int64_t>(max_output_length_ptr);
  ShapeVector output_shape = {Shape::SHP_ANY};
  ShapeVector min_shape = {1};
  ShapeVector max_shape = {max_output_length};
  ShapePtr shape = std::make_shared<Shape>(output_shape, min_shape, max_shape);

  return std::make_shared<AbstractTensor>(range_start_type, shape);
}

AbstractBasePtr InferImplArgMaxWithValue(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  // check keep_dims
  ValuePtr keep_dims = primitive->GetAttr("keep_dims");
  MS_EXCEPTION_IF_NULL(keep_dims);
  if (!keep_dims->isa<BoolImm>()) {
    MS_LOG(EXCEPTION) << "keep_dims should be Bool.";
  }
  bool keep_dims_value = GetValue<bool>(keep_dims);
  // check axis
  ValuePtr axis = primitive->GetAttr("axis");
  MS_EXCEPTION_IF_NULL(axis);
  if (!axis->isa<Int32Imm>() && !axis->isa<Int64Imm>()) {
    MS_LOG(EXCEPTION) << "axis should be Int.";
  }
  // check axis convert negative to positive value
  auto check_axis = [](int64_t &axis, const size_t dim) -> void {
    auto dim_ = static_cast<int64_t>(dim);
    if (axis < -dim_ || axis >= dim_) {
      MS_LOG(EXCEPTION) << "axis should be in [" << -dim_ << ", " << dim_ << "). But got axis = " << axis << ".";
    }
    if (axis >= -dim_ && axis < 0) {
      axis += dim_;
    }
    return;
  };
  // main calculate shape func
  auto cal_shape = [axis, keep_dims_value, check_axis](ShapeVector &shape, const ShapeVector &x_shape) -> void {
    (void)shape.insert(shape.end(), x_shape.begin(), x_shape.end());
    auto axis_value = GetValue<int64_t>(axis);
    check_axis(axis_value, x_shape.size());
    if (keep_dims_value) {
      shape[axis_value] = 1;
    } else {
      (void)shape.erase(std::begin(shape) + axis_value);
    }
  };
  ShapeVector shape = {};
  ShapeVector min_shape = {};
  ShapeVector max_shape = {};
  ShapeVector x_shape = x->shape()->shape();
  ShapeVector x_min_shape = x->shape()->min_shape();
  ShapeVector x_max_shape = x->shape()->max_shape();
  CheckMinMaxShape(x_shape, &x_min_shape, &x_max_shape);
  cal_shape(shape, x_shape);
  cal_shape(min_shape, x_min_shape);
  cal_shape(max_shape, x_max_shape);
  TypePtr idx_type = kInt32;
  auto index = std::make_shared<AbstractTensor>(idx_type, std::make_shared<Shape>(shape, min_shape, max_shape));
  auto value = std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
  AbstractBasePtrList result = {index, value};
  return std::make_shared<AbstractTuple>(result);
}

AbstractBasePtr InferImplSort(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list) {
  const std::string &op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractTensorPtr input = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);

  TypePtrList supported_types = {kFloat16, kFloat32};
  (void)CheckTensorDType(input, supported_types, "input for Sort should be %s");

  ValuePtr axis_ptr = primitive->GetAttr("axis");
  int64_t axis = GetValue<int64_t>(axis_ptr);
  int64_t input_rank = input->shape()->shape().size();
  if (input_rank == 0) {
    MS_LOG(EXCEPTION) << "input must be a Tensor with dimension > 0.";
  }

  if (!(axis >= -input_rank && axis < input_rank)) {
    MS_LOG(EXCEPTION) << "axis is not in the valid range [" << -input_rank << ", " << input_rank << ").";
  }

  auto sorted_values = std::make_shared<AbstractTensor>(input->element(), input->shape());
  TypePtr idx_type = kInt32;
  auto indices = std::make_shared<AbstractTensor>(idx_type, input->shape());
  AbstractBasePtrList result = {sorted_values, indices};
  return std::make_shared<AbstractTuple>(result);
}

AbstractBasePtr InferImplMaskedSelect(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  const size_t size_expected = 2;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  AbstractTensorPtr x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  AbstractTensorPtr mask = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);

  auto x_shape = x->shape();
  auto mask_shape = mask->shape();
  auto broadcast_shape = BroadcastShape(x_shape->shape(), mask_shape->shape());
  ShapeVector y_shape = {Shape::SHP_ANY};
  ShapeVector min_shape = {1};
  int64_t max_size = std::accumulate(broadcast_shape.begin(), broadcast_shape.end(), 1, std::multiplies<int64_t>());
  ShapeVector max_shape = {max_size};
  if (max_shape.empty()) {
    max_shape = x_shape->shape();
  }
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(y_shape, min_shape, max_shape));
}

AbstractBasePtr InferImplDynamicStitch(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(primitive);
  bool output_shape_unknow = false;
  auto prim_name = primitive->name();
  constexpr int64_t args_size = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(args_spec_list.size()), kEqual, args_size,
                                           prim_name);
  for (const auto &item : args_spec_list) {
    MS_EXCEPTION_IF_NULL(item);
  }

  // input0: indices
  auto input_tuple = args_spec_list[0]->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(input_tuple);
  auto indices = input_tuple->elements();
  auto input_indice_size = input_tuple->size();
  int64_t first_dim_size = 0;
  for (size_t i = 0; i < input_indice_size; i++) {
    auto indicei = indices[i]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(indicei);
    auto valuei = indicei->BuildValue();
    MS_EXCEPTION_IF_NULL(valuei);
    if (!valuei->isa<tensor::Tensor>()) {
      output_shape_unknow = true;
      continue;
    }
    auto indicei_value = CheckAndConvertUtils::CheckTensorIntValue("indices", valuei, prim_name);
    auto indicei_max = std::max_element(indicei_value.begin(), indicei_value.end());
    first_dim_size = *indicei_max > first_dim_size ? *indicei_max : first_dim_size;
  }

  auto indices0 = indices[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(indices0);
  auto indices0_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices0->BuildShape())[kShape];

  // input1: data
  auto input_tuple_1 = args_spec_list[1]->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(input_tuple_1);
  auto data = input_tuple_1->elements();
  auto data0 = data[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(data0);
  auto data0_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(data0->BuildShape())[kShape];
  if (indices.size() != data.size()) {
    MS_LOG(EXCEPTION) << "The number of input[0] must be the same as input[0]!";
  }

  int64_t indices_total_size = 0;
  std::map<std::string, TypePtr> types;
  (void)types.emplace("data0", data0->BuildType());
  for (size_t i = 1; i < data.size(); ++i) {
    auto indicesi_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices[i]->BuildShape())[kShape];
    auto datai_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(data[i]->BuildShape())[kShape];
    if (indicesi_shape.size() > datai_shape.size()) {
      MS_LOG(EXCEPTION) << "The rank of indices[i] must be <= rank of data[i]!";
    }
    indices_total_size += SizeToLong(indicesi_shape.size());
  }
  std::set<TypePtr> valid_types = ops::common_valid_types;
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);

  ShapeVector out_shape;
  if (output_shape_unknow) {
    out_shape.push_back(abstract::Shape::SHP_ANY);
  } else {
    out_shape.push_back(first_dim_size + 1);
  }
  for (size_t i = indices0_shape.size(); i < data0_shape.size(); ++i) {
    out_shape.push_back(data0_shape[i]);
  }
  ShapeVector min_shape = out_shape;
  ShapeVector max_shape = out_shape;
  if (output_shape_unknow) {
    // delete after dynamic alloc is support
    const int64_t EXPAND_MAX = 10;
    min_shape = out_shape;
    max_shape = out_shape;
    min_shape[0] = 1;
    max_shape[0] = indices_total_size * EXPAND_MAX;
  }
  return std::make_shared<AbstractTensor>(infer_type,
                                          std::make_shared<abstract::Shape>(out_shape, min_shape, max_shape));
}

AbstractBasePtr InferImplTensorCopySlices(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const AbstractBasePtrList &args_spec_list) {
  auto &op_name = primitive->name();
  constexpr auto kTensorCopySlicesInputNum = 5;
  CheckArgsSize(op_name, args_spec_list, kTensorCopySlicesInputNum);
  AbstractTensorPtr input = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  return std::make_shared<AbstractTensor>(input->element(), input->shape());
}
}  // namespace abstract
}  // namespace luojianet_ms
