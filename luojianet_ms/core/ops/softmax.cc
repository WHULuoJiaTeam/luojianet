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

#include "ops/softmax.h"
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
void Softmax::set_axis(const std::vector<int64_t> &axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }

std::vector<int64_t> Softmax::get_axis() const {
  auto value_ptr = GetAttr(kAxis);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Softmax::Init(const int64_t axis) {
  auto op_name = this->name();
  std::vector<int64_t> axis_vec = {axis};
  (void)CheckAndConvertUtils::CheckInteger("axis_len", SizeToLong(axis_vec.size()), kEqual, 1, op_name);
  auto rank = SizeToLong(axis_vec.size());
  for (auto &item : axis_vec) {
    CheckAndConvertUtils::CheckInRange<int64_t>("axis", item, kIncludeLeft, {-rank, rank}, op_name);
  }
  this->set_axis(axis_vec);
}

namespace {
abstract::ShapePtr SoftMaxInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto axis = GetValue<std::vector<int64_t>>(primitive->GetAttr(kAxis));
  (void)CheckAndConvertUtils::CheckValue<size_t>("length of axis", axis.size(), kGreaterEqual, 1, op_name);
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  if (shape_map.empty()) {
    // Scalar input, has no shape
    return std::make_shared<abstract::Shape>(std::vector<int64_t>());
  }
  auto in_shape = shape_map[kShape];
  auto min_shape = shape_map[kMinShape];
  auto max_shape = shape_map[kMaxShape];
  auto rank = SizeToLong(in_shape.size());
  for (auto &item : axis) {
    CheckAndConvertUtils::CheckInRange<int64_t>("axis", item, kIncludeLeft, {-rank, rank}, op_name);
  }
  if (min_shape.size() != 0 && max_shape.size() != 0) {
    return std::make_shared<abstract::Shape>(in_shape, min_shape, max_shape);
  }
  return std::make_shared<abstract::Shape>(in_shape);
}

TypePtr SoftMaxInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr &a) { return a == nullptr; })) {
    MS_LOG(EXCEPTION) << "For '" << prim->name()
                      << ", the input args used for infer shape and type is necessary, but missing it.";
  }
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  return CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, prim->name());
}
}  // namespace

MIND_API_BASE_IMPL(Softmax, PrimitiveC, BaseOperator);
AbstractBasePtr SoftmaxInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(SoftMaxInferShape(primitive, input_args), SoftMaxInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(Softmax, prim::kPrimSoftmax, SoftmaxInfer, nullptr, true);
}  // namespace ops
}  // namespace luojianet_ms
