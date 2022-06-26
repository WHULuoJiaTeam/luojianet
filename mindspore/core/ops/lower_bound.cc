/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/lower_bound.h"
#include "ops/op_utils.h"
#include "abstract/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr LowerBoundInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  size_t size_exp = 2;
  if (x_shape.size() != size_exp) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', The rank of sorted_x need to be equal to 2, but got " << values_shape.size();
  }
  if (values_shape.size() != size_exp) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', The rank of values need to be equal to 2, but got "
                             << values_shape.size();
  }
  if (x_shape[0] != values_shape[0]) {
    MS_EXCEPTION(ValueError)
      << "For '" << primitive->name()
      << "', The first dimension of shape of sorted_x must be equal to that of values, but got shape of values: "
      << input_args[1]->BuildShape()->ToString() << ", shape of sorted_x:" << input_args[0]->BuildShape()->ToString()
      << ".";
  }
  return std::make_shared<abstract::Shape>(values_shape);
}

TypePtr LowerBoundInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  std::map<std::string, TypePtr> input_types;
  std::set<TypePtr> input_valid_types = {kFloat16, kFloat32, kFloat64, kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16};
  TypePtr sorted_x_type = input_args[0]->BuildType();
  TypePtr values_type = input_args[1]->BuildType();
  (void)input_types.emplace("sorted_x", sorted_x_type);
  (void)input_types.emplace("values", values_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(input_types, input_valid_types, primitive->name());
  auto dtype_attr = primitive->GetAttr("out_type");
  auto out_type = dtype_attr->cast<TypePtr>();
  auto out_type_id = out_type->type_id();
  MS_EXCEPTION_IF_NULL(out_type);
  if (out_type_id != kInt32->type_id() && out_type_id != kInt64->type_id()) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', out_type must be int32 or int64, but got "
                            << out_type;
  }
  return out_type;
}
}  // namespace

MIND_API_BASE_IMPL(LowerBound, PrimitiveC, BaseOperator);
AbstractBasePtr LowerBoundInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = LowerBoundInferType(primitive, input_args);
  auto infer_shape = LowerBoundInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(LowerBound, prim::kPrimLowerBound, LowerBoundInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
