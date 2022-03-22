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

#include "ops/real.h"
#include <map>
#include <string>
#include <set>
#include "abstract/primitive_infer_map.h"
#include "ops/op_utils.h"

namespace luojianet_ms {
namespace ops {
namespace {
abstract::ShapePtr RealInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto in_shape = shape_map[kShape];
  auto min_shape = shape_map[kMinShape];
  auto max_shape = shape_map[kMaxShape];
  return std::make_shared<abstract::Shape>(in_shape, min_shape, max_shape);
}

TypePtr RealInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto input_type = input_args[kInputIndex0]->BuildType();
  const std::set<TypePtr> all_types_with_complex = {kBool,    kInt,     kInt8,    kInt16,     kInt32,     kInt64,
                                                    kUInt,    kUInt8,   kUInt16,  kUInt32,    kUInt64,    kFloat,
                                                    kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", input_type, all_types_with_complex, prim->name());
  auto input_tensor = input_type->cast<TensorTypePtr>();
  TypeId input_tensor_id = input_tensor->element()->type_id();
  if (input_tensor_id == kNumberTypeComplex64) {
    return kFloat32;
  }
  if (input_tensor_id == kNumberTypeComplex128) {
    return kFloat64;
  }
  return input_type;
}

AbstractBasePtr RealInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());

  return abstract::MakeAbstract(RealInferShape(primitive, input_args), RealInferType(primitive, input_args));
}
}  // namespace
REGISTER_PRIMITIVE_EVAL_IMPL(Real, prim::kPrimReal, RealInfer, nullptr, true);
}  // namespace ops
}  // namespace luojianet_ms
