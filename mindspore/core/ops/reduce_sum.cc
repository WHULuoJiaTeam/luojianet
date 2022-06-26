/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <memory>
#include <algorithm>
#include <set>

#include "ops/reduce_sum.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
int64_t InferImplReduceFuncCheckAxis(const PrimitivePtr &prim, const int64_t &axis, const size_t dim) {
  int64_t dim_ = static_cast<int64_t>(dim);
  if (axis < -dim_ || axis >= dim_) {
    MS_LOG(EXCEPTION) << "For '" << prim->name() << "', axis should be in [" << -dim_ << ", " << dim_
                      << "). But got axis = " << axis;
  }
  int64_t ret_axis = axis;
  if (axis >= -dim_ && axis < 0) {
    ret_axis += dim_;
  }
  return ret_axis;
}

void InferImplReduceFuncCalShape(const PrimitivePtr &primitive, ShapeVector *shape, const ShapeVector &x_shape,
                                 const ValuePtr &axis, bool keep_dims_value) {
  MS_EXCEPTION_IF_NULL(axis);
  MS_EXCEPTION_IF_NULL(shape);
  if (axis->isa<ValueTuple>() || axis->isa<ValueList>()) {
    auto axis_ptr_list =
      axis->isa<ValueTuple>() ? axis->cast<ValueTuplePtr>()->value() : axis->cast<ValueListPtr>()->value();
    if (!axis_ptr_list.size()) {
      if (keep_dims_value) (void)shape->insert(shape->end(), x_shape.size(), 1);
    } else {
      (void)shape->insert(shape->end(), x_shape.begin(), x_shape.end());
      ValuePtrList axis_items = axis_ptr_list;
      ValuePtrList::iterator it;
      if (keep_dims_value) {
        for (it = axis_items.begin(); it != axis_items.end(); ++it) {
          auto axis_value = InferImplReduceFuncCheckAxis(primitive, GetValue<int64_t>(*it), x_shape.size());
          shape->at(LongToSize(axis_value)) = 1;
        }
      } else {
        std::vector<int64_t> axis_value_list;
        for (it = axis_items.begin(); it != axis_items.end(); ++it) {
          auto axis_value = GetValue<int64_t>(*it);
          auto axis_positive_value = InferImplReduceFuncCheckAxis(primitive, axis_value, x_shape.size());
          axis_value_list.push_back(axis_positive_value);
        }
        std::sort(axis_value_list.begin(), axis_value_list.end());
        std::vector<int64_t>::reverse_iterator it_re;
        for (it_re = axis_value_list.rbegin(); it_re != axis_value_list.rend(); ++it_re) {
          (void)shape->erase(shape->begin() + *it_re);
        }
      }
    }
  } else if (axis->isa<Int32Imm>() || axis->isa<Int64Imm>()) {
    (void)shape->insert(shape->end(), x_shape.begin(), x_shape.end());
    int64_t axis_value = GetValue<int64_t>(axis);
    axis_value = InferImplReduceFuncCheckAxis(primitive, axis_value, x_shape.size());
    if (keep_dims_value) {
      shape->at(axis_value) = 1;
    } else {
      (void)shape->erase(shape->begin() + axis_value);
    }
  } else {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', Axis should be one of types: [int/tuple/list].";
  }
  return;
}

bool CheckShapeValid(const std::vector<AbstractBasePtr> &input_args, const uint64_t input_num_ascend) {
  return input_args.size() == input_num_ascend && input_args[1] && input_args[1]->isa<abstract::AbstractTensor>() &&
         input_args[1]->BuildValue() && input_args[1]->BuildValue()->isa<AnyValue>();
}

abstract::ShapePtr ReduceSumInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto shape_ptr = CheckAndConvertUtils::GetTensorInputShape("ReduceSum", input_args, 0);
  MS_EXCEPTION_IF_NULL(shape_ptr);
  auto input_shape = shape_ptr->shape();
  auto input_min_shape = shape_ptr->min_shape();
  auto input_max_shape = shape_ptr->max_shape();
  auto keep_dimis_value_ptr = primitive->GetAttr(kKeepDims);
  MS_EXCEPTION_IF_NULL(keep_dimis_value_ptr);
  if (!keep_dimis_value_ptr->isa<BoolImm>()) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', keep_dims should be Bool.";
  }
  bool keep_dims = GetValue<bool>(keep_dimis_value_ptr);
  ShapeVector out_shape = {};
  ShapeVector out_min_shape = {};
  ShapeVector out_max_shape = {};
  int64_t max_v;
  if (shape_ptr->IsDynamic()) {
    max_v = *max_element(input_max_shape.begin(), input_max_shape.end());
  } else {
    max_v = *max_element(input_shape.begin(), input_shape.end());
  }
  const uint64_t input_num_ascend = 2;
  if (CheckShapeValid(input_args, input_num_ascend)) {
    auto axis_tensor = input_args[1]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(axis_tensor);
    auto axis_tensor_shape = axis_tensor->shape();
    MS_EXCEPTION_IF_NULL(axis_tensor_shape);
    auto axis_shape = axis_tensor_shape->shape();
    if (axis_shape.size() == 1 && axis_shape[0] == -1 && !keep_dims) {
      out_shape.push_back(-2);
      out_min_shape = input_min_shape;
      out_max_shape = input_max_shape;
    } else if (!keep_dims) {
      for (size_t i = 0; i < input_shape.size() - axis_shape.size(); ++i) {
        out_shape.push_back(-1);
        out_min_shape.push_back(1);
        out_max_shape.push_back(max_v);
      }
    } else {
      for (size_t i = 0; i < input_shape.size(); ++i) {
        out_shape.push_back(-1);
        out_min_shape.push_back(1);
        out_max_shape.push_back(max_v);
      }
    }
    return std::make_shared<abstract::Shape>(out_shape, out_min_shape, out_max_shape);
  } else {
    ValuePtr axis_value;
    ValuePtr axis_ptr;
    if (input_args.size() == input_num_ascend && input_args[1]) {
      axis_ptr = input_args[1]->BuildValue();
    } else {
      axis_ptr = primitive->GetAttr("axis");
    }
    MS_EXCEPTION_IF_NULL(axis_ptr);
    if (axis_ptr->isa<tensor::Tensor>() && input_args[1]) {
      auto axis_type = input_args[1]->BuildType();
      MS_EXCEPTION_IF_NULL(axis_type);
      auto axis_type_id = axis_type->cast<TensorTypePtr>();
      MS_EXCEPTION_IF_NULL(axis_type_id);
      auto axis_tensor = axis_ptr->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(axis_tensor);
      size_t data_size = axis_tensor->DataSize();
      std::vector<ValuePtr> value_list;
      auto element = axis_type_id->element();
      MS_EXCEPTION_IF_NULL(element);
      if (element->type_id() == kNumberTypeInt32) {
        auto shape_data = reinterpret_cast<int *>(axis_tensor->data_c());
        MS_EXCEPTION_IF_NULL(shape_data);
        for (size_t i = 0; i < data_size; i++) {
          value_list.push_back(MakeValue(static_cast<int64_t>(*shape_data)));
          ++shape_data;
        }
      } else {
        auto shape_data2 = reinterpret_cast<int64_t *>(axis_tensor->data_c());
        for (size_t i = 0; i < data_size; i++) {
          value_list.push_back(MakeValue(static_cast<int64_t>(*shape_data2)));
          ++shape_data2;
        }
      }
      axis_value = std::make_shared<ValueTuple>(value_list);
    } else {
      axis_value = axis_ptr;
    }
    InferImplReduceFuncCalShape(primitive, &out_shape, input_shape, axis_value, keep_dims);

    if (!input_min_shape.empty() && !input_max_shape.empty()) {
      ShapeVector shape_min = {};
      ShapeVector shape_max = {};
      InferImplReduceFuncCalShape(primitive, &shape_min, input_min_shape, axis_value, keep_dims);
      InferImplReduceFuncCalShape(primitive, &shape_max, input_max_shape, axis_value, keep_dims);
      return std::make_shared<abstract::Shape>(out_shape, shape_min, shape_max);
    }
    return std::make_shared<abstract::Shape>(out_shape);
  }
}

TypePtr ReduceSumInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto x_type = input_args[0]->BuildType();
  std::set<TypePtr> valid_types = common_valid_types;
  valid_types.insert(kBool);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x dtype", x_type, valid_types, prim->name());
  return x_type;
}
}  // namespace

MIND_API_BASE_IMPL(ReduceSum, PrimitiveC, Reduce);
AbstractBasePtr ReduceSumInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  const int64_t input_num = 1;
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInteger("input size", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                     primitive->name());
  return abstract::MakeAbstract(ReduceSumInferShape(primitive, input_args), ReduceSumInferType(primitive, input_args));
}
}  // namespace ops
}  // namespace mindspore
