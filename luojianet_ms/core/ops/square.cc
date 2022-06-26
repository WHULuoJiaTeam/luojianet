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

#include "ops/square.h"
#include "abstract/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
namespace {
template <typename T>
void ImpleSquare(void *origin, void *target, size_t size) {
  MS_EXCEPTION_IF_NULL(origin);
  MS_EXCEPTION_IF_NULL(target);
  auto origin_data = reinterpret_cast<T *>(origin);
  auto target_data = reinterpret_cast<T *>(target);
  for (size_t i = 0; i < size; ++i) {
    target_data[i] = origin_data[i] * origin_data[i];
  }
}

abstract::ShapePtr SquareInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto in_shape = shape_map[kShape];
  auto min_shape = shape_map[kMinShape];
  auto max_shape = shape_map[kMaxShape];
  return std::make_shared<abstract::Shape>(in_shape, min_shape, max_shape);
}

TypePtr SquareInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto x_dtype = input_args[kInputIndex0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_dtype);
  if (!x_dtype->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "For Primitive['Square'], the input argument['x'] "
                               "must be a Tensor but got "
                            << x_dtype->ToString() << ".";
  }
  return x_dtype;
}

AbstractBasePtr SquareInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto types = SquareInferType(primitive, input_args);
  auto shapes = SquareInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

ValuePtr SquareInferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  if (input_args.empty()) {
    return nullptr;
  }
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x = input_args[kInputIndex0]->BuildValue();
  if (x == nullptr) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(x);
  auto x_tensor = x->cast<tensor::TensorPtr>();
  if (x_tensor == nullptr) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(x_tensor);
  auto data_size = x_tensor->DataSize();
  auto dtype = x_tensor->data_type();
  auto infer_shape = SquareInferShape(prim, input_args);
  MS_EXCEPTION_IF_NULL(infer_shape);
  auto shape = infer_shape->shape();
  auto result_tensor = std::make_shared<tensor::Tensor>(dtype, shape);  // same shape and dtype
  auto x_datac = x_tensor->data_c();
  MS_EXCEPTION_IF_NULL(result_tensor);
  auto result_datac = result_tensor->data_c();
  switch (dtype) {
    case kNumberTypeInt8: {
      ImpleSquare<int8_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeInt16: {
      ImpleSquare<int16_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeInt32: {
      ImpleSquare<int32_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeInt64: {
      ImpleSquare<int64_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeUInt8: {
      ImpleSquare<uint8_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeUInt16: {
      ImpleSquare<uint16_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeUInt32: {
      ImpleSquare<uint32_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeUInt64: {
      ImpleSquare<uint64_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeFloat16: {
      ImpleSquare<float16>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeFloat32: {
      ImpleSquare<float>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeFloat64: {
      ImpleSquare<double>(x_datac, result_datac, data_size);
      break;
    }
    default: {
      MS_EXCEPTION(TypeError) << "For '" << prim->name()
                              << "', the supported data type is ['int8', 'int16', 'int32', 'int64', 'uint8', "
                                 "'uint16','uint32', 'uint64','float16', 'float32', 'float64'], but got "
                              << x_tensor->ToString();
    }
  }
  return result_tensor;
}
}  // namespace

MIND_API_BASE_IMPL(Square, PrimitiveC, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(Square, prim::kPrimSquare, SquareInfer, SquareInferValue, true);
}  // namespace ops
}  // namespace luojianet_ms
