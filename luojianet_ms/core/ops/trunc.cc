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

#include "ops/trunc.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
namespace {
abstract::ShapePtr TruncInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto x_shape = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x_shape);
  auto output_shape = x_shape->cast<abstract::ShapePtr>();
  return output_shape;
}

TypePtr TruncInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  CheckAndConvertUtils::CheckInteger("input number", input_args.size(), kEqual, 1, prim->name());
  std::set<TypePtr> check_list = {kFloat16, kFloat32, kInt8, kInt32, kUInt8};
  auto input_type = input_args[0]->BuildType();
  CheckAndConvertUtils::CheckTensorTypeValid("input_x", input_type, check_list, prim->name());
  return input_type;
}
}  // namespace

MIND_API_BASE_IMPL(Trunc, PrimitiveC, BaseOperator);
AbstractBasePtr TruncInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(TruncInferShape(primitive, input_args), TruncInferType(primitive, input_args));
}

REGISTER_PRIMITIVE_EVAL_IMPL(Trunc, prim::kPrimTrunc, TruncInfer, nullptr, true);
}  // namespace ops
}  // namespace luojianet_ms
