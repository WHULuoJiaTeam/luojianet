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
#include <memory>
#include "ops/crop_and_resize.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
MIND_API_BASE_IMPL(CropAndResize, PrimitiveC, BaseOperator);
void CropAndResize::Init(ResizeMethod method, float extrapolation_value) {
  this->set_method(method);
  this->set_extrapolation_value(extrapolation_value);
}

void CropAndResize::set_method(ResizeMethod method) {
  auto swi = (int64_t)method;
  (void)this->AddAttr(kMethod, api::MakeValue(swi));
}

void CropAndResize::set_extrapolation_value(float extrapolation_value) {
  (void)this->AddAttr(kExtrapolationValue, api::MakeValue(extrapolation_value));
}

ResizeMethod CropAndResize::get_method() const {
  auto value_ptr = GetAttr(kMethod);
  return ResizeMethod(GetValue<int64_t>(value_ptr));
}

float CropAndResize::get_extrapolation_value() const {
  auto value_ptr = GetAttr(kExtrapolationValue);
  return GetValue<float>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameCropAndResize, CropAndResize);
}  // namespace ops
}  // namespace luojianet_ms
