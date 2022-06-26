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

#include <map>
#include <string>
#include <set>
#include <vector>
#include <memory>

#include "ops/logical_not.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
namespace {
abstract::ShapePtr LogicalNotInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto in_shape = shape_map[kShape];
  auto min_shape = shape_map[kMinShape];
  auto max_shape = shape_map[kMaxShape];
  return std::make_shared<abstract::Shape>(in_shape, min_shape, max_shape);
}

TypePtr LogicalNotInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto op_name = prim->name();
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto infer_dtype = input_args[0]->BuildType();
  std::set<TypePtr> local_bool = {kBool};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", infer_dtype, local_bool, op_name);
  return infer_dtype;
}
}  // namespace

MIND_API_BASE_IMPL(LogicalNot, PrimitiveC, BaseOperator);
AbstractBasePtr LogicalNotInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, primitive->name());
  auto infer_type = LogicalNotInferType(primitive, input_args);
  auto infer_shape = LogicalNotInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(LogicalNot, prim::kPrimLogicalNot, LogicalNotInfer, nullptr, true);
}  // namespace ops
}  // namespace luojianet_ms
