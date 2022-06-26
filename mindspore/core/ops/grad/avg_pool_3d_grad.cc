/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "ops/grad/avg_pool_3d_grad.h"
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
abstract::ShapePtr AvgPool3DGradInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input size", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->GetShapeTrack())[kShape];
  constexpr int64_t k5DInputDims = 5;
  (void)CheckAndConvertUtils::CheckInteger("grad_rank", SizeToLong(grad_shape.size()), kEqual, k5DInputDims, op_name);
  std::vector<int64_t> origin_input_size;
  if (input_args[0]->isa<abstract::AbstractTuple>()) {  // origin_size is tuple
    origin_input_size = GetValue<std::vector<int64_t>>(input_args[0]->BuildValue());
  } else {
    MS_LOG(EXCEPTION) << "For '" << op_name << "', the first input data size must be a tuple.";
  }
  return std::make_shared<abstract::Shape>(origin_input_size);
}

TypePtr AvgPool3DGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input size", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto grad_dtype = input_args[1]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  return CheckAndConvertUtils::CheckTensorTypeValid("grad", grad_dtype, valid_types, op_name);
}
}  // namespace

MIND_API_BASE_IMPL(AvgPool3DGrad, PrimitiveC, BaseOperator);
AbstractBasePtr AvgPool3DGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  auto res = std::make_shared<abstract::AbstractTensor>(AvgPool3DGradInferType(primitive, input_args),
                                                        AvgPool3DGradInferShape(primitive, input_args)->shape());
  return res;
}

REGISTER_PRIMITIVE_EVAL_IMPL(AvgPool3DGrad, prim::kPrimAvgPool3DGrad, AvgPool3DGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
