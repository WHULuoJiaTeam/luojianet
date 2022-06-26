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
#include "ops/xlogy.h"
#include <functional>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <set>
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"
#include "ops/primitive_c.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
namespace {
abstract::ShapePtr XlogyInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  return BroadCastInferShape(prim_name, input_args);
}
TypePtr XlogyInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto y = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 1);
  (void)abstract::CheckDtypeSame(prim_name, x, y);
  auto input_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(input_type);
  const std::set<TypePtr> valid_types = {kFloat32, kFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("y", input_type, valid_types, prim_name);
  return input_type;
}
}  // namespace

MIND_API_BASE_IMPL(Xlogy, PrimitiveC, BaseOperator);
AbstractBasePtr XlogyInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto type = XlogyInferType(primitive, input_args);
  auto shape = XlogyInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Xlogy, prim::kPrimXlogy, XlogyInfer, nullptr, true);
}  // namespace ops
}  // namespace luojianet_ms
