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

#include <memory>

#include "ops/fused_batch_norm.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
MIND_API_BASE_IMPL(FusedBatchNorm, PrimitiveC, BaseOperator);
void FusedBatchNorm::Init(const int64_t mode, const float epsilon, const float momentum) {
  this->set_mode(mode);
  this->set_epsilon(epsilon);
  this->set_momentum(momentum);
}

void FusedBatchNorm::set_mode(const int64_t mode) { (void)this->AddAttr(kMode, api::MakeValue(mode)); }

void FusedBatchNorm::set_epsilon(const float epsilon) { (void)this->AddAttr(kEpsilon, api::MakeValue(epsilon)); }

void FusedBatchNorm::set_momentum(const float momentum) { (void)this->AddAttr(kMomentum, api::MakeValue(momentum)); }

int64_t FusedBatchNorm::get_mode() const {
  auto value_ptr = this->GetAttr(kMode);
  return GetValue<int64_t>(value_ptr);
}

float FusedBatchNorm::get_epsilon() const {
  auto value_ptr = this->GetAttr(kEpsilon);
  return GetValue<float>(value_ptr);
}

float FusedBatchNorm::get_momentum() const {
  auto value_ptr = this->GetAttr(kMomentum);
  return GetValue<float>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameFusedBatchNorm, FusedBatchNorm);
}  // namespace ops
}  // namespace luojianet_ms
