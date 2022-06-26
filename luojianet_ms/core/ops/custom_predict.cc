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

#include "ops/custom_predict.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
MIND_API_BASE_IMPL(CustomPredict, PrimitiveC, BaseOperator);
void CustomPredict::Init(const int64_t output_num, const float weight_threshold) {
  this->set_output_num(output_num);
  this->set_weight_threshold(weight_threshold);
}

void CustomPredict::set_output_num(const int64_t output_num) {
  (void)this->AddAttr(kOutputNum, api::MakeValue(output_num));
}

int64_t CustomPredict::get_output_num() const {
  auto value_ptr = this->GetAttr(kOutputNum);
  return GetValue<int64_t>(value_ptr);
}

void CustomPredict::set_weight_threshold(const float weight_threshold) {
  (void)this->AddAttr(kWeightThreshold, api::MakeValue(weight_threshold));
}

float CustomPredict::get_weight_threshold() const {
  auto value_ptr = this->GetAttr(kWeightThreshold);
  return GetValue<float>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameCustomPredict, CustomPredict);
}  // namespace ops
}  // namespace luojianet_ms
