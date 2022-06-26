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
#include "ops/bn_training_reduce.h"

#include <map>
#include <set>
#include <vector>
#include <string>

#include "ops/op_utils.h"
#include "utils/tensor_construct_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
namespace {
int64_t BNTrainingReduceGetAndCheckFormat(const PrimitivePtr &primitive, const ValuePtr &value) {
  int64_t data_format;
  bool result = CheckAndConvertUtils::GetDataFormatEnumValue(value, &data_format);
  if (!result || (data_format != Format::NHWC && data_format != Format::NCHW && data_format != Format::NCDHW)) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', data format should be NCHW, NHWC or NCDHW, but got "
                      << data_format << ".";
  }
  return data_format;
}
abstract::TupleShapePtr BNTrainingReduceInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto shape = input_shape[kShape];
  auto min_shape = input_shape[kMinShape];
  auto max_shape = input_shape[kMaxShape];

  constexpr auto kInputDim = 4;
  (void)CheckAndConvertUtils::CheckInteger("x_dim", SizeToLong(shape.size()), kEqual, kInputDim, primitive->name());
  auto data_format_ptr = primitive->GetAttr("format");
  MS_EXCEPTION_IF_NULL(data_format_ptr);
  int64_t data_format = BNTrainingReduceGetAndCheckFormat(primitive, data_format_ptr);
  size_t c_axis = kInputIndex1;
  if (data_format == Format::NHWC) {
    c_axis = kInputIndex3;
  }
  ShapeVector batch = {shape[c_axis]};
  abstract::ShapePtr sum_shape;
  abstract::ShapePtr square_sum_shape;
  if (min_shape.empty() || max_shape.empty()) {
    sum_shape = std::make_shared<abstract::Shape>(batch);
    square_sum_shape = std::make_shared<abstract::Shape>(batch);
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{sum_shape, square_sum_shape});
  }

  ShapeVector batch_min = {min_shape[c_axis]};
  ShapeVector batch_max = {max_shape[c_axis]};
  sum_shape = std::make_shared<abstract::Shape>(batch, batch_min, batch_max);
  square_sum_shape = std::make_shared<abstract::Shape>(batch, batch_min, batch_max);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{sum_shape, square_sum_shape});
}

TypePtr BNTrainingReduceInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto input_type = input_args[0]->BuildType();
  std::set<TypePtr> check_list = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_type, check_list, primitive->name());
  return std::make_shared<Tuple>(std::vector<TypePtr>{input_type, input_type});
}
}  // namespace

MIND_API_BASE_IMPL(BNTrainingReduce, PrimitiveC, BaseOperator);
AbstractBasePtr BNTrainingReduceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t kInputNum = 1;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto infer_type = BNTrainingReduceInferType(primitive, input_args);
  auto infer_shape = BNTrainingReduceInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(BNTrainingReduce, prim::kPrimBNTrainingReduce, BNTrainingReduceInfer, nullptr, true);
}  // namespace ops
}  // namespace luojianet_ms
