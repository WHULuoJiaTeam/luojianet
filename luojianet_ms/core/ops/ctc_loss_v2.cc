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

#include "ops/ctc_loss_v2.h"
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
namespace {
abstract::TupleShapePtr CTCLossV2InferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  constexpr size_t kLenLogProbs = 3;
  constexpr size_t kLenTarget = 2;
  constexpr int64_t kMulti = 2;
  constexpr int64_t kInputSize = 4;
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kInputSize,
                                           prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto log_probs_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto targets_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  auto log_probs_shape = log_probs_shape_map[kShape];
  auto targets_shape = targets_shape_map[kShape];
  if (log_probs_shape.size() != kLenLogProbs) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', Input log_probs's dims must be 3, but got :" << log_probs_shape.size();
  }
  if (targets_shape.size() != kLenTarget) {
    MS_LOG(EXCEPTION) << "For '" << prim_name << "', Input targets's dims must be 2, but got :" << targets_shape.size();
  }
  int64_t T = log_probs_shape[0];
  int64_t N = log_probs_shape[1];
  int64_t S = targets_shape[1];

  std::vector<int64_t> out_dim0 = {N};
  std::vector<int64_t> out_dim1 = {N, T, kMulti * S + 1};
  abstract::ShapePtr neg_log_shape = std::make_shared<abstract::Shape>(out_dim0);
  abstract::ShapePtr log_alpha_shape = std::make_shared<abstract::Shape>(out_dim1);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{neg_log_shape, log_alpha_shape});
}

TuplePtr CTCLossV2InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto name = primitive->name();
  const std::set<TypePtr> valid_types = {kFloat32};
  auto type = CheckAndConvertUtils::CheckTypeValid("log_probs", input_args[0]->BuildType(), valid_types, name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{type, type});
}
}  // namespace

MIND_API_BASE_IMPL(CTCLossV2, PrimitiveC, BaseOperator);
AbstractBasePtr CTCLossV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  return abstract::MakeAbstract(CTCLossV2InferShape(primitive, input_args), CTCLossV2InferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(CTCLossV2, prim::kPrimCTCLossV2, CTCLossV2Infer, nullptr, true);
}  // namespace ops
}  // namespace luojianet_ms
