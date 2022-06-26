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

#include "ops/grad/hsigmoid_grad.h"

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr HSigmoidGradInferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                           primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto prim_name = primitive->name();
  auto grads_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto input_x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  CheckAndConvertUtils::Check("grads_shape", grads_shape, kEqual, input_x_shape, prim_name, TypeError);
  return std::make_shared<abstract::Shape>(grads_shape);
}

TypePtr HSigmoidGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim->name());
  const std::set<TypePtr> valid_types = {kInt8, kInt16, kInt32, kInt64, kFloat16, kFloat32};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("grads", input_args[0]->BuildType());
  (void)types.emplace("input_x", input_args[1]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
}
}  // namespace

MIND_API_BASE_IMPL(HSigmoidGrad, PrimitiveC, BaseOperator);
AbstractBasePtr HSigmoidGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = HSigmoidGradInferType(primitive, input_args);
  auto infer_shape = HSigmoidGradInferShape(primitive, input_args);
  MS_EXCEPTION_IF_NULL(infer_shape);
  return std::make_shared<abstract::AbstractTensor>(infer_type, infer_shape->shape());
}
}  // namespace ops
}  // namespace mindspore
