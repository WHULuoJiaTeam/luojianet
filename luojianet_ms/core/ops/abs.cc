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

#include "ops/abs.h"
#include <string>
#include <algorithm>
#include <set>
#include <map>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
namespace {
template <typename T>
void ImpleAbs(void *origin, void *target, size_t size) {
  MS_EXCEPTION_IF_NULL(origin);
  MS_EXCEPTION_IF_NULL(target);
  auto origin_data = reinterpret_cast<T *>(origin);
  auto target_data = reinterpret_cast<T *>(target);
  auto zero_val = static_cast<T>(0);
  for (size_t i = 0; i < size; ++i) {
    target_data[i] = origin_data[i] >= zero_val ? origin_data[i] : -origin_data[i];
  }
}

abstract::ShapePtr AbsInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, 1, prim_name);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto x = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr AbsInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto x_type = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, common_valid_types, prim->name());
  return x_type;
}

AbstractBasePtr AbsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());

  return abstract::MakeAbstract(AbsInferShape(primitive, input_args), AbsInferType(primitive, input_args));
}

ValuePtr AbsInferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  if (input_args.empty()) {
    return nullptr;
  }

  auto x = input_args[0]->BuildValue();
  if (x == nullptr) {
    return nullptr;
  }
  auto x_tensor = x->cast<tensor::TensorPtr>();
  if (x_tensor == nullptr) {
    return nullptr;
  }

  auto data_size = x_tensor->DataSize();
  auto dtype = x_tensor->data_type();
  auto shape = AbsInferShape(prim, input_args);
  auto result_tensor = std::make_shared<tensor::Tensor>(dtype, shape->shape());
  auto x_datac = x_tensor->data_c();
  auto result_datac = result_tensor->data_c();
  switch (dtype) {
    case kNumberTypeInt8: {
      ImpleAbs<int8_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeInt16: {
      ImpleAbs<int16_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeInt32: {
      ImpleAbs<int32_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeInt64: {
      ImpleAbs<int64_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeUInt8: {
      ImpleAbs<uint8_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeUInt16: {
      ImpleAbs<uint16_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeUInt32: {
      ImpleAbs<uint32_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeUInt64: {
      ImpleAbs<uint64_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeFloat16: {
      ImpleAbs<float16>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeFloat32: {
      ImpleAbs<float>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeFloat64: {
      ImpleAbs<double>(x_datac, result_datac, data_size);
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

MIND_API_BASE_IMPL(Abs, PrimitiveC, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(Abs, prim::kPrimAbs, AbsInfer, AbsInferValue, true);
}  // namespace ops
}  // namespace luojianet_ms
