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
#include "ops/truncate_div.h"
#include <algorithm>
#include <functional>
#include <string>
#include <vector>
#include <memory>
#include <set>
#include "abstract/abstract_value.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
namespace {
abstract::ShapePtr TruncateDivInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  return BroadCastInferShape(prim_name, input_args);
}

TypePtr TruncateDivInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[0]);
  MS_EXCEPTION_IF_NULL(input_args[1]);
  auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto y = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 1);
  (void)abstract::CheckDtypeSame(prim_name, x, y);
  auto z_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(z_type);
  if (!z_type->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "The " << prim_name << "'s"
                            << " input must be tensor type but got " << z_type->ToString();
  }
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
        << "For '" << prim_name
        << "', Complex math binary op expecting Tensor [complex64, complex64],[complex64, float32], [float32, "
           "complex64],[complex128, complex128],[complex128, float64], [float64, complex128], but got["
        << type_x->ToString() << ", " << type_y->ToString() << "].";
    }
  }
  const std::set<TypePtr> valid_types = {kNumber, kBool};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", type_x, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("y", type_y, valid_types, prim_name);
  return z_type;
}
}  // namespace

MIND_API_BASE_IMPL(TruncateDiv, PrimitiveC, BaseOperator);
AbstractBasePtr TruncateDivInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  auto type = TruncateDivInferType(primitive, input_args);
  auto shape = TruncateDivInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(TruncateDiv, prim::kPrimTruncateDiv, TruncateDivInfer, nullptr, true);
}  // namespace ops
}  // namespace luojianet_ms
