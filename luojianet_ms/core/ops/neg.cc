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

#include "ops/neg.h"
#include <complex>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
namespace {
template <typename T>
void ImpleNeg(void *origin, void *target, size_t size) {
  MS_EXCEPTION_IF_NULL(origin);
  MS_EXCEPTION_IF_NULL(target);
  auto origin_data = reinterpret_cast<T *>(origin);
  auto target_data = reinterpret_cast<T *>(target);
  for (size_t i = 0; i < size; ++i) {
    target_data[i] = -origin_data[i];
  }
}
abstract::ShapePtr NegInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto x = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_ptr = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  return shape_ptr;
}

TypePtr NegInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kUInt8,   kInt8,    kInt16,   kInt32,     kInt64,
                                         kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, prim->name());
  return input_args[0]->BuildType();
}

ValuePtr NegInferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  if (input_args.empty()) {
    return nullptr;
  }

  auto x = input_args[kInputIndex0]->BuildValue();
  if (x == nullptr) {
    return nullptr;
  }
  auto x_tensor = x->cast<tensor::TensorPtr>();
  if (x_tensor == nullptr) {
    return nullptr;
  }

  auto data_size = x_tensor->DataSize();
  auto dtype = x_tensor->data_type();
  auto shape = NegInferShape(prim, input_args)->shape();
  auto result_tensor = std::make_shared<tensor::Tensor>(dtype, shape);  // same shape and dtype
  auto x_datac = x_tensor->data_c();
  auto result_datac = result_tensor->data_c();
  switch (dtype) {
    case kNumberTypeInt8: {
      ImpleNeg<int8_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeInt16: {
      ImpleNeg<int16_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeInt32: {
      ImpleNeg<int32_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeInt64: {
      ImpleNeg<int64_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeUInt8: {
      ImpleNeg<uint8_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeUInt16: {
      ImpleNeg<uint16_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeUInt32: {
      ImpleNeg<uint32_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeUInt64: {
      ImpleNeg<uint64_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeFloat16: {
      ImpleNeg<float16>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeFloat32: {
      ImpleNeg<float>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeFloat64: {
      ImpleNeg<double>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeComplex64: {
      ImpleNeg<std::complex<float>>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeComplex128: {
      ImpleNeg<std::complex<double>>(x_datac, result_datac, data_size);
      break;
    }
    default: {
      MS_EXCEPTION(TypeError)
        << "For '" << prim->name()
        << "', the supported data type is ['int8', 'int16', 'int32', 'int64', 'uint8', "
           "'uint16','uint32', 'uint64','float16', 'float32', 'float64', 'complex64', 'complex128'], but got "
        << x_tensor->ToString();
    }
  }
  return result_tensor;
}
}  // namespace

MIND_API_BASE_IMPL(Neg, PrimitiveC, BaseOperator);
AbstractBasePtr NegInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = NegInferType(primitive, input_args);
  auto infer_shape = NegInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Neg, prim::kPrimNeg, NegInfer, NegInferValue, true);
}  // namespace ops
}  // namespace luojianet_ms
