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

#include "ops/dtype.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
MIND_API_BASE_IMPL(DType, PrimitiveC, BaseOperator);
ValuePtr DTypeInferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("dtype infer", int64_t(input_args.size()), kEqual, 1, op_name);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  if (type->isa<TensorType>()) {
    const std::set<TypePtr> valid_types = {kTensorType};
    return CheckAndConvertUtils::CheckTensorTypeValid("input_x", type, valid_types, op_name);
  } else if (type->isa<CSRTensorType>() || type->isa<COOTensorType>()) {
    const std::set<TypePtr> valid_types = {kCSRTensorType, kCOOTensorType};
    return CheckAndConvertUtils::CheckSparseTensorTypeValid("input_x", type, valid_types, op_name);
  }
  MS_EXCEPTION(TypeError) << "For Primitive[" << op_name << "], the input argument[input_x] "
                          << "must be a Tensor, CSRTensor or COOTensor, but got " << type->ToString() << ".";
  return nullptr;
}

AbstractBasePtr DTypeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto value = DTypeInferValue(primitive, input_args);
  MS_EXCEPTION_IF_NULL(value);
  auto type = value->cast<TypePtr>();
  MS_EXCEPTION_IF_NULL(type);
  return abstract::MakeAbstract(std::make_shared<abstract::NoShape>(), type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(DType, prim::kPrimDType, DTypeInfer, DTypeInferValue, false);
}  // namespace ops
}  // namespace luojianet_ms
