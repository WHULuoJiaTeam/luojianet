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
#include <algorithm>

#include "ops/greater.h"
#include "ops/op_utils.h"
#include "abstract/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
namespace {
abstract::ShapePtr GreaterInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  return BroadCastInferShape(op_name, input_args);
}

TypePtr GreaterInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const int64_t kInputNum = 2;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, prim->name());
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->BuildType());
  (void)types.emplace("y", input_args[1]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types_with_complex_and_bool, prim->name());
  return std::make_shared<TensorType>(kBool);
}
}  // namespace

MIND_API_BASE_IMPL(Greater, PrimitiveC, BaseOperator);
AbstractBasePtr GreaterInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = GreaterInferType(primitive, input_args);
  auto infer_shape = GreaterInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_C(kNameGreater, Greater);
}  // namespace ops
}  // namespace luojianet_ms
