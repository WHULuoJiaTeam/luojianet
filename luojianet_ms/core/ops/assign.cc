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

#include <set>
#include <map>
#include <vector>
#include <memory>
#include <string>

#include "ops/assign.h"
#include "ops/op_utils.h"
#include "ir/dtype/ref.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
MIND_API_BASE_IMPL(Assign, PrimitiveC, BaseOperator);
abstract::ShapePtr AssignInferShape(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  auto variable_shape_ptr = input_args[0]->BuildShape();
  auto value_shape_ptr = input_args[1]->BuildShape();
  auto variable_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(variable_shape_ptr)[kShape];
  auto value_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(value_shape_ptr)[kShape];
  auto shape_element = variable_shape_ptr->cast<abstract::ShapePtr>();
  if (variable_shape.size() != value_shape.size()) {
    if (variable_shape.size() == 1 && variable_shape[0] == 1 && value_shape.empty()) {
      return shape_element;
    } else if (value_shape.size() == 1 && value_shape[0] == 1 && variable_shape.empty()) {
      return shape_element;
    } else {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", the rank of value is " << value_shape.size()
                               << ". It should be same with variable's rank " << variable_shape.size() << ".";
    }
  }
  for (uint64_t i = 0; i < variable_shape.size(); i++) {
    if (variable_shape[i] != value_shape[i]) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", the shape of value is " << value_shape_ptr->ToString()
                               << ". It should be same with variable's shape " << variable_shape_ptr->ToString() << ".";
    }
  }
  return shape_element;
}

TypePtr AssignInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  auto check_types = common_valid_types;
  (void)check_types.emplace(kBool);
  auto value_type = input_args[1]->BuildType();
  auto variable_type = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("variable", variable_type, check_types, prim_name);
  CheckAndConvertUtils::CheckScalarOrTensorTypesSame(std::map<std::string, TypePtr>{{"value", value_type}}, check_types,
                                                     prim_name);
  return variable_type;
}

AbstractBasePtr AssignInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("infer", SizeToLong(CheckAndConvertUtils::GetRemoveMonadAbsNum(input_args)),
                                           kEqual, input_num, prim_name);
  auto variable_type = input_args[0]->BuildType();
  if (variable_type->isa<RefKeyType>()) {
    return input_args[1]->Broaden();
  }
  return abstract::MakeAbstract(AssignInferShape(primitive, input_args), AssignInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(Assign, prim::kPrimAssign, AssignInfer, nullptr, true);
}  // namespace ops
}  // namespace luojianet_ms
