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
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "ops/addn.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
namespace {
abstract::ShapePtr AddNInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto elements = input_args[0]->isa<abstract::AbstractTuple>()
                    ? input_args[0]->cast<abstract::AbstractTuplePtr>()->elements()
                    : input_args[0]->cast<abstract::AbstractListPtr>()->elements();
  (void)CheckAndConvertUtils::CheckInteger("concat element num", SizeToLong(elements.size()), kGreaterEqual, 1,
                                           primitive->name());
  (void)primitive->AddAttr("N", MakeValue(SizeToLong(elements.size())));
  (void)primitive->AddAttr("n", MakeValue(SizeToLong(elements.size())));
  auto shape_0 = elements[0]->BuildShape();
  auto element0_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(shape_0);
  for (size_t i = 0; i < elements.size(); ++i) {
    auto shape = elements[i]->BuildShape();
    if (shape->isa<abstract::Shape>() && shape_0->isa<abstract::Shape>()) {
      const auto &shape_vec = shape->cast<abstract::ShapePtr>()->shape();
      const auto &shape_0_vec = shape_0->cast<abstract::ShapePtr>()->shape();
      if ((shape_vec == ShapeVector({1}) && shape_0_vec == ShapeVector()) ||
          (shape_vec == ShapeVector() && shape_0_vec == ShapeVector({1}))) {
        MS_LOG(DEBUG) << "The primitive[" << primitive->name() << "]'s input[" << i << "] shape: " << shape->ToString()
                      << " are consistent with the shape of input[0]" << shape_0->ToString();
        continue;
      }
    }
    if (!shape->IsDynamic() && !shape_0->IsDynamic()) {
      if (*shape != *shape_0) {
        MS_EXCEPTION(ValueError) << "The primitive[" << primitive->name() << "]'s input shape must be same, "
                                 << "but got the shape of input[" << i << "]: " << shape->ToString()
                                 << ", shape of input[0]:" << shape_0->ToString();
      }
    }
  }
  auto in_shape = element0_shape_map[kShape];
  auto min_shape = element0_shape_map[kMinShape];
  auto max_shape = element0_shape_map[kMaxShape];
  return std::make_shared<abstract::Shape>(in_shape, min_shape, max_shape);
}

TypePtr AddNInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  auto elements = input_args[0]->isa<abstract::AbstractTuple>()
                    ? input_args[0]->cast<abstract::AbstractTuplePtr>()->elements()
                    : input_args[0]->cast<abstract::AbstractListPtr>()->elements();
  (void)CheckAndConvertUtils::CheckInteger("concat element num", SizeToLong(elements.size()), kGreaterEqual, 1,
                                           prim->name());
  std::map<std::string, TypePtr> types;
  (void)types.emplace("element_0", elements[0]->BuildType());
  for (size_t i = 0; i < elements.size(); ++i) {
    if (elements[i]->BuildType()->type_id() == kObjectTypeUndeterminedType) {
      return elements[0]->BuildType();
    }
    std::string element_i = "element_" + std::to_string(i);
    (void)types.emplace(element_i, elements[i]->BuildType());
  }
  std::set<TypePtr> valid_types = common_valid_types;
  valid_types.insert(kBool);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);
  return elements[0]->BuildType();
}
}  // namespace

MIND_API_BASE_IMPL(AddN, PrimitiveC, BaseOperator);
AbstractBasePtr AddNInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  if (!input_args[0]->isa<abstract::AbstractTuple>() && !input_args[0]->isa<abstract::AbstractList>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the input data type must be list or tuple of tensors.";
  }
  const int64_t kInputNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
  auto infer_type = AddNInferType(primitive, input_args);
  auto infer_shape = AddNInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(AddN, prim::kPrimAddN, AddNInfer, nullptr, true);
}  // namespace ops
}  // namespace luojianet_ms
