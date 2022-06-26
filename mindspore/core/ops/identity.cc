/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "ops/identity.h"

#include <string>
#include <vector>
#include <memory>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr IdentityInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  return CheckAndConvertUtils::GetTensorInputShape(op_name, input_args, 0);
}

TypePtr IdentityInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto infer_type = input_args[0]->BuildType();
  auto valid_type = common_valid_types_with_complex_and_bool;
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", infer_type, valid_type, op_name);
  return infer_type;
}
}  // namespace

MIND_API_BASE_IMPL(Identity, PrimitiveC, BaseOperator);
AbstractBasePtr IdentityInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto infer_type = IdentityInferType(primitive, input_args);
  auto infer_shape = IdentityInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Identity, prim::kPrimIdentitys, IdentityInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
