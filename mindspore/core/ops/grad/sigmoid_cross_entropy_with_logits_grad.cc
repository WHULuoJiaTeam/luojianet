/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ops/grad/sigmoid_cross_entropy_with_logits_grad.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SigmoidCrossEntropyWithLogitsGradInferShape(const PrimitivePtr &primitive,
                                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kInputNum = 3;
  (void)CheckAndConvertUtils::CheckInteger("sigmoid_cross_extropy_with_logits_infer_shape",
                                           SizeToLong(input_args.size()), kGreaterEqual, kInputNum, prim_name);
  auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
  auto y = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex1);
  auto dout = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex2);
  auto x_ptr = x->BuildShape()->cast<abstract::ShapePtr>();
  abstract::CheckShapeSame(prim_name, x, y);
  abstract::CheckShapeSame(prim_name, x, dout);
  MS_EXCEPTION_IF_NULL(x_ptr);
  return x_ptr;
}

TypePtr SigmoidCrossEntropyWithLogitsGradInferType(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kInputNum = 3;
  (void)CheckAndConvertUtils::CheckInteger("sigmoid_cross_extropy_with_logits_infer_type",
                                           SizeToLong(input_args.size()), kGreaterEqual, kInputNum, prim_name);
  auto x_type = input_args[0]->BuildType();
  auto y_type = input_args[1]->BuildType();
  auto dout_type = input_args[2]->BuildType();
  const std::set<TypePtr> valid_types = {kBool,   kInt,    kInt8,   kInt16, kInt32,   kInt64,   kUInt,    kUInt8,
                                         kUInt16, kUInt32, kUInt64, kFloat, kFloat16, kFloat32, kFloat64, kComplex64};
  std::map<std::string, TypePtr> args;
  (void)args.emplace("x_type", x_type);
  (void)args.emplace("y_type", y_type);
  (void)args.emplace("dout_type", dout_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, primitive->name());
  return dout_type;
}
}  // namespace

MIND_API_BASE_IMPL(SigmoidCrossEntropyWithLogitsGrad, PrimitiveC, BaseOperator);
AbstractBasePtr SigmoidCrossEntropyWithLogitsGradInfer(const abstract::AnalysisEnginePtr &,
                                                       const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = SigmoidCrossEntropyWithLogitsGradInferType(primitive, input_args);
  auto infer_shape = SigmoidCrossEntropyWithLogitsGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(SigmoidCrossEntropyWithLogitsGrad, prim::kPrimSigmoidCrossEntropyWithLogitsGrad,
                             SigmoidCrossEntropyWithLogitsGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
