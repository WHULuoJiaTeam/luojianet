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

#include "ops/grad/cdist_grad.h"

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
namespace {
abstract::ShapePtr CdistGradInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto y_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape())[kShape];
  auto cdist_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[3]->BuildShape())[kShape];
  auto x_size = x_shape.size();
  auto y_size = y_shape.size();
  CheckAndConvertUtils::Check("grad shape", grad_shape, kEqual, cdist_shape, prim_name, ValueError);
  if (x_size != y_size) {
    MS_EXCEPTION(ValueError) << "For 'CdistGrad', rank of input_x and input_y should be equal, but got input_x size: "
                             << x_size << ", input_y size: " << y_size << ".";
  }
  CheckAndConvertUtils::CheckInRange("input_x dim", x_size, kIncludeBoth, {2, 3}, "CdistGrad");
  auto out_shape = x_shape;
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr CdistGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> valid_types = {kFloat32, kFloat16};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("grad", input_args[0]->BuildType());
  (void)types.emplace("input_x", input_args[1]->BuildType());
  (void)types.emplace("input_y", input_args[2]->BuildType());
  (void)types.emplace("cdist", input_args[3]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
}
}  // namespace

MIND_API_BASE_IMPL(CdistGrad, PrimitiveC, BaseOperator);
AbstractBasePtr CdistGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = CdistGradInferType(primitive, input_args);
  auto infer_shape = CdistGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(CdistGrad, prim::kPrimCdistGrad, CdistGradInfer, nullptr, true);
}  // namespace ops
}  // namespace luojianet_ms
