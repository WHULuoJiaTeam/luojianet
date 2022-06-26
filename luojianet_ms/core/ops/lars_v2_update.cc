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

#include "ops/lars_v2_update.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
namespace {
abstract::ShapePtr LARSUpdateInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  MS_LOG(INFO) << "For '" << op_name << "', it's now doing infer shape.";
  const int64_t input_num = 6;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  auto weight_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack());
  auto gradient_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->GetShapeTrack());
  auto norm_weight_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->GetShapeTrack());
  auto norm_gradient_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[3]->GetShapeTrack());
  auto weight_decay_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[4]->GetShapeTrack());
  auto learning_rate_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[5]->GetShapeTrack());

  if (weight_shape[kShape].size() != gradient_shape[kShape].size()) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', weight shape size should be equal to gradient shape size, but got "
                             << "weight shape: " << weight_shape << " and gradient shape: " << gradient_shape;
  }
  if (norm_weight_shape[kShape].size() != norm_gradient_shape[kShape].size()) {
    MS_EXCEPTION(ValueError) << "For " << op_name
                             << "', norm weight shape size should be equal to norm gradient shape size, but got "
                             << "weight shape: " << norm_weight_shape << " and gradient shape: " << norm_gradient_shape;
  }
  for (size_t index = 0; index < weight_shape[kShape].size(); index++) {
    if (weight_shape[kShape][index] != gradient_shape[kShape][index]) {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', The " << index
                               << "'s shape  of weight shape should euqal with gradient shape, but got "
                               << "weight shape: " << norm_weight_shape
                               << " and gradient shape:" << norm_gradient_shape;
    }
  }
  for (size_t index = 0; index < weight_shape[kShape].size(); index++) {
    if (weight_shape[kShape][index] != gradient_shape[kShape][index]) {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', The " << index
                               << "'s shape  of weight shape should euqal with gradient shape, but got "
                               << "weight shape: " << norm_weight_shape
                               << " and gradient shape:" << norm_gradient_shape;
    }
  }
  auto shp_len = weight_decay_shape[kShape].size();
  auto para_name = input_args[4]->ToString();
  (void)CheckAndConvertUtils::CheckInteger(para_name, shp_len, kLessEqual, 1);
  if (shp_len == 1) {
    (void)CheckAndConvertUtils::CheckInteger(para_name, weight_decay_shape[kShape][0], kEqual, 1);
  }
  shp_len = learning_rate_shape[kShape].size();
  para_name = input_args[5]->ToString();
  (void)CheckAndConvertUtils::CheckInteger(para_name, shp_len, kLessEqual, 1);
  if (shp_len == 1) {
    (void)CheckAndConvertUtils::CheckInteger(para_name, learning_rate_shape[kShape][0], kEqual, 1);
  }

  return std::make_shared<abstract::Shape>(weight_shape[kShape], weight_shape[kMinShape], weight_shape[kMaxShape]);
}

TypePtr LARSUpdateInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 6;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  std::map<std::string, TypePtr> types;
  (void)types.emplace("Weight dtype", input_args[0]->BuildType());
  (void)types.emplace("gradient dtype", input_args[1]->BuildType());
  (void)types.emplace("norm weight dtype", input_args[2]->BuildType());
  (void)types.emplace("norm gradient dtype", input_args[3]->BuildType());
  const std::set<TypePtr> valid_types = {kInt16, kInt32, kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(types, valid_types, primitive->name(), true);
  return types["Weight dtype"];
}
}  // namespace

MIND_API_BASE_IMPL(LARSUpdate, PrimitiveC, BaseOperator);
AbstractBasePtr LARSUpdateInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = LARSUpdateInferType(primitive, input_args);
  auto infer_shape = LARSUpdateInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(LARSUpdate, prim::kPrimLARSUpdate, LARSUpdateInfer, nullptr, true);
}  // namespace ops
}  // namespace luojianet_ms
