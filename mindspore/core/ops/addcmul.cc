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

#include <map>
#include <set>
#include <string>
#include <algorithm>
#include "ops/addcmul.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr AddcmulInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_data_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto input_data = input_data_map[kShape];
  auto x1_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape());
  auto x1_shape = x1_shape_map[kShape];
  auto x2_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape());
  auto x2_shape = x2_shape_map[kShape];
  auto value_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape());
  auto value_shape = value_shape_map[kShape];
  auto broadcast_shape = CalBroadCastShape(x1_shape, x2_shape, op_name, "x1", "x2");
  if (input_args[kInputIndex3]->isa<abstract::AbstractTensor>()) {
    CalBroadCastShape(x1_shape, value_shape, op_name, "x1", "value");
    CalBroadCastShape(x2_shape, value_shape, op_name, "x2", "value");
    broadcast_shape = CalBroadCastShape(broadcast_shape, value_shape, op_name);
  }
  broadcast_shape = CalBroadCastShape(broadcast_shape, input_data, op_name);
  return std::make_shared<abstract::Shape>(broadcast_shape);
}

TypePtr AddcmulInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto op_name = prim->name();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kInt32};
  auto input_data_type = input_args[kInputIndex0]->BuildType();
  auto x1_type = input_args[kInputIndex1]->BuildType();
  auto x2_type = input_args[kInputIndex2]->BuildType();
  auto value_type = input_args[kInputIndex3]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_data", input_data_type, valid_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x1", x1_type, valid_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x2", x2_type, valid_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("value", value_type, valid_types, op_name);
  std::map<std::string, TypePtr> types;
  (void)types.emplace("input_data", input_data_type);
  (void)types.emplace("x1", x1_type);
  (void)types.emplace("x2", x2_type);
  (void)types.emplace("value", value_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  return input_data_type;
}
}  // namespace

MIND_API_BASE_IMPL(Addcmul, PrimitiveC, BaseOperator);
AbstractBasePtr AddcmulInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_shape = AddcmulInferShape(primitive, input_args);
  auto infer_type = AddcmulInferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Addcmul, prim::kPrimAddcmul, AddcmulInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
