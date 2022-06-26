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

#include "ops/transpose.h"
#include <vector>
#include <memory>
#include <algorithm>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
namespace {
abstract::ShapePtr TransposeInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x_min_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kMinShape];
  auto x_max_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kMaxShape];
  ShapeVector p_value;
  ShapeVector p_value_raw;
  if (input_args.size() == 1) {
    if (!primitive->HasAttr("perm")) {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', the value of input_perm is necessary, but missing it!";
    }
    ValuePtr perm = primitive->GetAttr("perm");
    MS_EXCEPTION_IF_NULL(perm);
    auto perm_val = perm->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(perm_val);
    auto perm_val_data = perm_val->value();
    (void)std::transform(std::begin(perm_val_data), std::end(perm_val_data), std::back_inserter(p_value_raw),
                         [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });
  } else {
    auto perm_value = input_args[1]->BuildValue();
    MS_EXCEPTION_IF_NULL(perm_value);
    if (perm_value->isa<tensor::Tensor>()) {
      p_value_raw = CheckAndConvertUtils::CheckTensorIntValue("perm", perm_value, op_name);
    } else {
      p_value_raw = CheckAndConvertUtils::CheckTupleInt("input[perm]", perm_value, op_name);
    }
  }
  for (auto p : p_value_raw) {
    p = (p >= 0) ? p : (p_value_raw.size() + p);
    p_value.emplace_back(p);
  }
  if (x_shape.size() != p_value.size()) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', The dimension of x and perm must be equal, but got x dimension: " << x_shape.size()
                             << ", perm dimension: " << p_value.size() << ".";
  }
  for (auto i : p_value) {
    (void)CheckAndConvertUtils::CheckInteger("perm element", i, kLessThan, SizeToLong(p_value.size()), op_name);
  }
  std::vector<int64_t> tmp(p_value);
  for (auto it = tmp.begin(); it != tmp.end();) {
    auto dim = *it;
    if (!tmp.empty()) {
      it = tmp.erase(it);
    }
    if (std::find(tmp.begin(), tmp.end(), dim) != tmp.end()) {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', The value of perm is wrong";
    }
  }
  std::vector<int64_t> in_shape(p_value);
  (void)std::transform(in_shape.begin(), in_shape.end(), in_shape.begin(), [x_shape](size_t i) { return x_shape[i]; });
  if (!x_min_shape.empty() && !x_max_shape.empty()) {
    std::vector<int64_t> min_shape;
    std::vector<int64_t> max_shape;
    for (auto i : p_value) {
      min_shape.push_back(x_min_shape[LongToSize(i)]);
      max_shape.push_back(x_max_shape[LongToSize(i)]);
    }
    return std::make_shared<abstract::Shape>(in_shape, min_shape, max_shape);
  } else {
    return std::make_shared<abstract::Shape>(in_shape);
  }
}

TypePtr TransposeInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  return CheckAndConvertUtils::CheckSubClass("x", input_args[0]->BuildType(), {kTensorType}, prim->name());
}
}  // namespace

MIND_API_BASE_IMPL(Transpose, PrimitiveC, BaseOperator);
AbstractBasePtr TransposeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  // The second input is optional.
  constexpr size_t input_size1 = 1;
  (void)CheckAndConvertUtils::CheckInteger("Transpose infer", SizeToLong(input_args.size()), kGreaterEqual, input_size1,
                                           primitive->name());
  auto type = TransposeInferType(primitive, input_args);
  auto shape = TransposeInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Transpose, prim::kPrimTranspose, TransposeInfer, nullptr, true);
}  // namespace ops
}  // namespace luojianet_ms
