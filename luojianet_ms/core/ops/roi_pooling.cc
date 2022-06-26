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
#include "ops/roi_pooling.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
MIND_API_BASE_IMPL(ROIPooling, PrimitiveC, BaseOperator);
void ROIPooling::set_pooled_h(const int64_t pooled_h) { (void)this->AddAttr(kPooledH, api::MakeValue(pooled_h)); }

int64_t ROIPooling::get_pooled_h() const { return GetValue<int64_t>(GetAttr(kPooledH)); }

void ROIPooling::set_pooled_w(const int64_t pooled_w) { (void)this->AddAttr(kPooledW, api::MakeValue(pooled_w)); }

int64_t ROIPooling::get_pooled_w() const {
  auto value_ptr = GetAttr(kPooledW);
  return GetValue<int64_t>(value_ptr);
}

void ROIPooling::set_scale(const float scale) { (void)this->AddAttr(kScale, api::MakeValue(scale)); }

float ROIPooling::get_scale() const {
  auto value_ptr = GetAttr(kScale);
  return GetValue<float>(value_ptr);
}

void ROIPooling::Init(const int64_t pooled_h, const int64_t pooled_w, const float scale) {
  this->set_pooled_h(pooled_h);
  this->set_pooled_w(pooled_w);
  this->set_scale(scale);
}

REGISTER_PRIMITIVE_C(kNameROIPooling, ROIPooling);
}  // namespace ops
}  // namespace luojianet_ms
