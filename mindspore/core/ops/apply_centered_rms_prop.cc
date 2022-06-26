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

#include "ops/apply_centered_rms_prop.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ApplyCenteredRMSPropInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  MS_LOG(INFO) << "For '" << op_name << "', it's now doing infer shape.";
  const int64_t kInputNum = 9;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, op_name);
  auto var_shape = input_args[0]->BuildShape();
  auto mg_shape = input_args[1]->BuildShape();
  auto ms_shape = input_args[2]->BuildShape();
  auto mom_shape = input_args[3]->BuildShape();
  auto grad_shape = input_args[4]->BuildShape();
  auto var_shape_ptr = var_shape->cast<abstract::ShapePtr>();
  auto mg_shape_ptr = mg_shape->cast<abstract::ShapePtr>();
  auto ms_shape_ptr = ms_shape->cast<abstract::ShapePtr>();
  auto mom_shape_ptr = mom_shape->cast<abstract::ShapePtr>();
  auto grad_shape_ptr = grad_shape->cast<abstract::ShapePtr>();
  // var and mg must have the same shape when is not dynamic
  if (!var_shape_ptr->IsDynamic() && !mg_shape_ptr->IsDynamic()) {
    if (*var_shape != *mg_shape) {
      MS_EXCEPTION(ValueError) << op_name << " mean gradient shape " << mg_shape->ToString()
                               << " are not consistent with var shape " << var_shape->ToString();
    }
  }
  // var and ms must have the same shape when is not dynamic
  if (!var_shape_ptr->IsDynamic() && !ms_shape_ptr->IsDynamic()) {
    if (*var_shape != *ms_shape) {
      MS_EXCEPTION(ValueError) << op_name << " mean square shape " << ms_shape->ToString()
                               << " are not consistent with var shape " << var_shape->ToString();
    }
  }
  // var and mom must have the same shape when is not dynamic
  if (!var_shape_ptr->IsDynamic() && !mom_shape_ptr->IsDynamic()) {
    if (*var_shape != *mom_shape) {
      MS_EXCEPTION(ValueError) << op_name << " moment shape " << mom_shape->ToString()
                               << " are not consistent with var shape " << var_shape->ToString();
    }
  }
  // var and grad must have the same shape when is not dynamic
  if (!var_shape_ptr->IsDynamic() && !grad_shape_ptr->IsDynamic()) {
    if (*var_shape != *grad_shape) {
      MS_EXCEPTION(ValueError) << op_name << " grad shape " << grad_shape->ToString()
                               << " are not consistent with var shape " << var_shape->ToString();
    }
  }
  auto shape_element = var_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr ApplyCenteredRMSPropInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 9;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto var_dtype = input_args[0]->BuildType();
  auto mean_gradient_dtype = input_args[1]->BuildType();
  auto mean_square_dtype = input_args[2]->BuildType();
  auto moment_dtype = input_args[3]->BuildType();
  auto grad_dtype = input_args[4]->BuildType();
  auto learning_rate_dtype = input_args[5]->BuildType();
  auto decay_dtype = input_args[6]->BuildType();
  auto momentum_dtype = input_args[7]->BuildType();
  auto epsilon_dtype = input_args[8]->BuildType();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("var dtype", var_dtype);
  (void)types.emplace("mean gradient dtype", mean_gradient_dtype);
  (void)types.emplace("mean square dtype", mean_square_dtype);
  (void)types.emplace("moment dtype", moment_dtype);
  (void)types.emplace("grad dtype", grad_dtype);
  const std::set<TypePtr> number_type = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,   kUInt32,
                                         kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex64};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, number_type, primitive->name());
  std::map<std::string, TypePtr> types_decay;
  (void)types_decay.emplace("decay dtype", decay_dtype);
  (void)types_decay.emplace("momentum dtype", momentum_dtype);
  (void)types_decay.emplace("epsilon dtype", epsilon_dtype);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(types_decay, valid_types, primitive->name());
  std::map<std::string, TypePtr> types_lr;
  (void)types_lr.emplace("learning rate dtype", learning_rate_dtype);
  (void)types_lr.emplace("decay dtype", decay_dtype);
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(types_lr, valid_types, primitive->name(), true);
  return var_dtype;
}
}  // namespace

MIND_API_BASE_IMPL(ApplyCenteredRMSProp, PrimitiveC, BaseOperator);
AbstractBasePtr ApplyCenteredRMSPropInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = ApplyCenteredRMSPropInferType(primitive, input_args);
  auto infer_shape = ApplyCenteredRMSPropInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(ApplyCenteredRMSProp, prim::kPrimApplyCenteredRMSProp, ApplyCenteredRMSPropInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
