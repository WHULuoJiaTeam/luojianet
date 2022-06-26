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

#include "ops/grad/bn_grad.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
MIND_API_BASE_IMPL(BNGrad, PrimitiveC, BaseOperator);
void BNGrad::Init(const float eps, const float momentum) {
  this->set_eps(eps);
  this->set_momentum(momentum);
}

void BNGrad::set_eps(const float eps) { (void)this->AddAttr(kEps, api::MakeValue(eps)); }

float BNGrad::get_eps() const {
  auto value_ptr = this->GetAttr(kEps);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<float>(value_ptr);
}

void BNGrad::set_momentum(const float momentum) { (void)this->AddAttr(kMomentum, api::MakeValue(momentum)); }

float BNGrad::get_momentum() const {
  auto value_ptr = this->GetAttr(kMomentum);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<float>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameBNGrad, BNGrad);
}  // namespace ops
}  // namespace luojianet_ms
