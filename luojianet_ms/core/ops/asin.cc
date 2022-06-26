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

#include "ops/asin.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "abstract/param_validator.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
namespace {
abstract::ShapePtr AsinInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto x = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr AsinInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  auto x_type = input_args[kInputIndex0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);
  return input_args[kInputIndex0]->BuildType();
}
}  // namespace

MIND_API_BASE_IMPL(Asin, PrimitiveC, BaseOperator);
AbstractBasePtr AsinInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const size_t InputNum = 1;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, InputNum, prim_name);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto types = AsinInferType(primitive, input_args);
  auto shapes = AsinInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

REGISTER_PRIMITIVE_EVAL_IMPL(Asin, prim::kPrimAsin, AsinInfer, nullptr, true);
}  // namespace ops
}  // namespace luojianet_ms
