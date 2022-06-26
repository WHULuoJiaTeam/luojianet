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
#include "ops/cholesky_inverse_.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr CholeskyInverseInferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kDimNum = 2;
  auto op_name = primitive->name();
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto x_shape = x_shape_map[kShape];
  if (x_shape.size() != kDimNum) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', The dimension of x must be equal to 2, but got : " << x_shape.size() << ".";
  }
  if (x_shape[x_shape.size() - 1] != x_shape[x_shape.size() - kDimNum])
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', input should be square matrix, "
                             << "while row is " << x_shape[x_shape.size() - kDimNum] << ", col is "
                             << x_shape[x_shape.size() - 1];
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr CholeskyInverseInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64};
  auto out_type =
    CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, prim->name());
  return out_type;
}
}  // namespace

MIND_API_BASE_IMPL(CholeskyInverse, PrimitiveC, BaseOperator);
AbstractBasePtr CholeskyInverseInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = CholeskyInverseInferType(primitive, input_args);
  auto infer_shape = CholeskyInverseInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(CholeskyInverse, prim::kPrimCholeskyInverse, CholeskyInverseInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
