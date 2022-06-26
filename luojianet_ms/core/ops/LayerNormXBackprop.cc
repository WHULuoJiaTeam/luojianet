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

#include "ops/LayerNormXBackprop.h"
#include <memory>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
namespace {
abstract::ShapePtr LayerNormXBackpropInferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x_shape = CheckAndConvertUtils::GetTensorInputShape(primitive->name(), input_args, 1);
  return x_shape;
}

TypePtr LayerNormXBackpropInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  // check
  std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  const int64_t x_index = 1;
  auto x_type = input_args[x_index]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);
  return x_type;
}
}  // namespace

MIND_API_BASE_IMPL(LayerNormXBackprop, PrimitiveC, BaseOperator);
AbstractBasePtr LayerNormXBackpropInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const int64_t input_num = 5;
  (void)CheckAndConvertUtils::CheckInteger("LayerNormXBackprop infer", SizeToLong(input_args.size()), kGreaterEqual,
                                           input_num, primitive->name());
  return abstract::MakeAbstract(LayerNormXBackpropInferShape(primitive, input_args),
                                LayerNormXBackpropInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(LayerNormXBackprop, prim::kPrimLayerNormXBackprop, LayerNormXBackpropInfer, nullptr, true);
}  // namespace ops
}  // namespace luojianet_ms
