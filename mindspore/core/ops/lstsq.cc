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

#include <memory>
#include <set>
#include <string>

#include "ops/lstsq.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr LstsqInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t x_dim_num = 2;
  const int64_t a_dim_num_1 = 1;
  const int64_t a_dim_num_2 = 2;

  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto x_shape = x_shape_map[kShape];
  auto a_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  auto a_shape = a_shape_map[kShape];
  if (x_shape.size() != x_dim_num) {
    MS_EXCEPTION(ValueError) << "For lstsq, the dimension of x must be equal to 2, while got x_dim: " << x_shape.size()
                             << ".";
  }
  if (a_shape.size() != a_dim_num_2 && a_shape.size() != a_dim_num_1) {
    MS_EXCEPTION(ValueError) << "For lstsq, the dimension of a must be equal to 2 or 1, while got a_dim: "
                             << a_shape.size() << ".";
  }
  if (x_shape[0] != a_shape[0]) {
    MS_EXCEPTION(ValueError) << "For lstsq, the length of x_dim[0]: " << x_shape[0]
                             << " is not equal to the length of a_dims[0]: " << a_shape[0] << ".";
  }
  ShapeVector y_shape;
  if (a_shape.size() == a_dim_num_1) {
    y_shape.push_back(x_shape[1]);
    y_shape.push_back(1);
  } else {
    y_shape.push_back(x_shape[1]);
    y_shape.push_back(a_shape[1]);
  }
  return std::make_shared<abstract::Shape>(y_shape);
}

TypePtr LstsqInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->BuildType());
  (void)types.emplace("a", input_args[1]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
}
}  // namespace

MIND_API_BASE_IMPL(Lstsq, PrimitiveC, BaseOperator);
AbstractBasePtr LstsqInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  auto infer_type = LstsqInferType(primitive, input_args);
  auto infer_shape = LstsqInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Lstsq, prim::kPrimLstsq, LstsqInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
