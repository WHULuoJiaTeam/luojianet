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
#include "ops/grad/sigmoid_grad.h"
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include "abstract/param_validator.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SigmoidGradInfershape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr SigmoidGradInfertype(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto y = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto dy = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 1);
  (void)abstract::CheckDtypeSame(prim_name, y, dy);
  auto x_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  const std::set<TypePtr> input_types = common_valid_types_with_complex;
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, input_types, prim->name());
  return x_type;
}
}  // namespace

MIND_API_BASE_IMPL(SigmoidGrad, PrimitiveC, BaseOperator);
AbstractBasePtr SigmoidGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto type = SigmoidGradInfertype(primitive, input_args);
  auto shape = SigmoidGradInfershape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(SigmoidGrad, prim::kPrimSigmoidGrad, SigmoidGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
