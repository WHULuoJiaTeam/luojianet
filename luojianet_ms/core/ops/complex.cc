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

#include "ops/complex.h"
#include <complex>
#include <map>
#include <string>
#include <set>
#include "abstract/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
namespace {
template <typename T>
void ImpleComplex(void *real, void *imag, void *target, size_t size) {
  MS_EXCEPTION_IF_NULL(real);
  MS_EXCEPTION_IF_NULL(imag);
  MS_EXCEPTION_IF_NULL(target);
  auto real_data = reinterpret_cast<T *>(real);
  auto imag_data = reinterpret_cast<T *>(imag);
  auto target_data = reinterpret_cast<std::complex<T> *>(target);
  MS_EXCEPTION_IF_NULL(real_data);
  MS_EXCEPTION_IF_NULL(imag_data);
  MS_EXCEPTION_IF_NULL(target_data);
  for (size_t i = 0; i < size; ++i) {
    target_data[i] = std::complex<T>(real_data[i], imag_data[i]);
  }
}

abstract::ShapePtr ComplexInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto in_shape = shape_map[kShape];
  auto min_shape = shape_map[kMinShape];
  auto max_shape = shape_map[kMaxShape];
  return std::make_shared<abstract::Shape>(in_shape, min_shape, max_shape);
}

TypePtr ComplexInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  std::map<std::string, TypePtr> types;
  auto real_input_type = input_args[kInputIndex0]->BuildType();
  auto imag_input_type = input_args[kInputIndex1]->BuildType();
  (void)types.emplace("real_input", real_input_type);
  (void)types.emplace("imag_input", imag_input_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, std::set<TypePtr>{kFloat32, kFloat64}, prim->name());
  auto real_input_tensor = real_input_type->cast<TensorTypePtr>();
  TypeId real_input_tensor_id = real_input_tensor->element()->type_id();
  return real_input_tensor_id == kNumberTypeFloat32 ? kComplex64 : kComplex128;
}

AbstractBasePtr ComplexInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());

  return abstract::MakeAbstract(ComplexInferShape(primitive, input_args), ComplexInferType(primitive, input_args));
}

ValuePtr ComplexInferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  if (input_args.empty()) {
    return nullptr;
  }

  auto real = input_args[kInputIndex0]->BuildValue();
  if (real == nullptr) {
    return nullptr;
  }
  auto real_tensor = real->cast<tensor::TensorPtr>();
  if (real_tensor == nullptr) {
    return nullptr;
  }

  auto imag = input_args[kInputIndex1]->BuildValue();
  if (imag == nullptr) {
    return nullptr;
  }
  auto imag_tensor = imag->cast<tensor::TensorPtr>();
  if (imag_tensor == nullptr) {
    return nullptr;
  }

  if (real_tensor->data_type() != imag_tensor->data_type()) {
    MS_EXCEPTION(TypeError) << "Inputs of Complex should be same, but got " << real_tensor->data_type() << "and "
                            << imag_tensor->data_type();
  }

  auto data_size = real_tensor->DataSize();
  auto dtype = real_tensor->data_type();
  auto shape = ComplexInferShape(prim, input_args)->shape();
  auto output_type = (dtype == kNumberTypeFloat32 ? kNumberTypeComplex64 : kNumberTypeComplex128);
  auto result_tensor = std::make_shared<tensor::Tensor>(output_type, shape);
  auto real_datac = real_tensor->data_c();
  auto imag_datac = imag_tensor->data_c();
  auto result_datac = result_tensor->data_c();
  switch (dtype) {
    case kNumberTypeFloat32: {
      ImpleComplex<float>(real_datac, imag_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeFloat64: {
      ImpleComplex<double>(real_datac, imag_datac, result_datac, data_size);
      break;
    }
    default: {
      MS_EXCEPTION(TypeError)
        << "For '" << prim->name()
        << "', the supported data type is in the list: ['kNumberTypeFloat32', 'kNumberTypeFloat64'], but got "
        << real_tensor->ToString() << ".";
    }
  }
  return result_tensor;
}
}  // namespace

MIND_API_BASE_IMPL(Complex, PrimitiveC, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(Complex, prim::kPrimComplex, ComplexInfer, ComplexInferValue, true);
}  // namespace ops
}  // namespace luojianet_ms
