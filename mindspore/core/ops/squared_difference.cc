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

#include <set>
#include <map>
#include <string>
#include "ops/squared_difference.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SquaredDifferenceInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  return BroadCastInferShape(op_name, input_args);
}

TypePtr SquaredDifferenceInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->BuildType());
  (void)types.emplace("y", input_args[1]->BuildType());
  const std::set<TypePtr> valid_types = {kInt32, kFloat16, kFloat32, kFloat64};
  auto type_x = input_args[0]->BuildType();
  auto type_y = input_args[1]->BuildType();
  MS_EXCEPTION_IF_NULL(type_x);
  MS_EXCEPTION_IF_NULL(type_y);
  if (type_x->isa<Complex>() || type_y->isa<Complex>()) {
    if (type_x->type_id() == kNumberTypeComplex64 && type_y->type_id() == kNumberTypeComplex64) {
      return type_x;
    } else if (type_x->type_id() == kNumberTypeComplex64 && type_y->type_id() == kNumberTypeFloat32) {
      return type_x;
    } else if (type_x->type_id() == kNumberTypeComplex128 && type_y->type_id() == kNumberTypeComplex128) {
      return type_x;
    } else if (type_x->type_id() == kNumberTypeComplex128 && type_y->type_id() == kNumberTypeFloat64) {
      return type_x;
    } else if (type_x->type_id() == kNumberTypeFloat32 && type_y->type_id() == kNumberTypeComplex64) {
      return type_y;
    } else if (type_x->type_id() == kNumberTypeFloat64 && type_y->type_id() == kNumberTypeComplex128) {
      return type_y;
    } else {
      MS_EXCEPTION(TypeError)
        << "For '" << prim->name()
        << "', Complex math binary op expecting Tensor [complex64, complex64],[complex64, float32], [float32, "
           "complex64],[complex128, complex128],[complex128, float64], [float64, complex128], but got["
        << type_x->ToString() << ", " << type_y->ToString() << "].";
    }
  }
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
}
}  // namespace

MIND_API_BASE_IMPL(SquaredDifference, PrimitiveC, BaseOperator);
AbstractBasePtr SquaredDifferenceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto infer_type = SquaredDifferenceInferType(primitive, input_args);
  auto infer_shape = SquaredDifferenceInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(SquaredDifference, prim::kPrimSquaredDifference, SquaredDifferenceInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
