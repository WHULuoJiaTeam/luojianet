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
#include "ops/grid_sampler_3d.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
namespace {
abstract::ShapePtr GridSampler3DInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  auto input_x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto grid_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  const size_t kFive = 5;
  if (input_x_shape.size() != kFive) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', Input_x must be a 5-dimensional tensor, but got "
                             << std::to_string(input_x_shape.size()) << "-dimensional tensor.";
  }
  if (grid_shape.size() != kFive) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', Grid must be a 5-dimensional tensor, but got "
                             << std::to_string(grid_shape.size()) << "-dimensional tensor.";
  }
  if (input_x_shape[kInputIndex0] != grid_shape[kInputIndex0]) {
    MS_EXCEPTION(ValueError)
      << "For '" << primitive->name()
      << "', The first dimension of 'grid' and 'input_x' must be equal, but got the shape of 'grid' is "
      << input_args[kInputIndex1]->BuildShape()->ToString() << " , and the shape of 'input_x' is "
      << input_args[kInputIndex0]->BuildShape()->ToString() << ".";
  }
  if (grid_shape[kInputIndex4] != kInputIndex3) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', The last dimension of grid must be 3, but got "
                             << std::to_string(grid_shape[kInputIndex4]);
  }
  std::vector<int64_t> output_shape = {input_x_shape[kInputIndex0], input_x_shape[kInputIndex1],
                                       grid_shape[kInputIndex1], grid_shape[kInputIndex2], grid_shape[kInputIndex3]};
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr GridSampler3DInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  std::map<std::string, TypePtr> types;
  std::set<TypePtr> valid_types = {kFloat32, kFloat64};
  TypePtr input_x_type = input_args[kInputIndex0]->BuildType();
  TypePtr grid_type = input_args[kInputIndex1]->BuildType();
  (void)types.emplace("input_x", input_x_type);
  (void)types.emplace("grid", grid_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  return input_x_type;
}
}  // namespace

MIND_API_BASE_IMPL(GridSampler3D, PrimitiveC, BaseOperator);
AbstractBasePtr GridSampler3DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = kInputIndex2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = GridSampler3DInferType(primitive, input_args);
  auto infer_shape = GridSampler3DInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(GridSampler3D, prim::kPrimGridSampler3D, GridSampler3DInfer, nullptr, true);
}  // namespace ops
}  // namespace luojianet_ms
