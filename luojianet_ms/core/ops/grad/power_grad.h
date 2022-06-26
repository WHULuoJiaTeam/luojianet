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

#ifndef LUOJIANET_MS_CORE_OPS_POWER_GRAD_H_
#define LUOJIANET_MS_CORE_OPS_POWER_GRAD_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace luojianet_ms {
namespace ops {
constexpr auto kNamePowerGrad = "PowerGrad";
class MIND_API PowerGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(PowerGrad);
  PowerGrad() : BaseOperator(kNamePowerGrad) {}
  void Init(const float power, const float scale, const float shift);
  void set_power(const float power);
  void set_scale(const float scale);
  void set_shift(const float shift);
  float get_power() const;
  float get_scale() const;
  float get_shift() const;
};
}  // namespace ops
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CORE_OPS_POWER_GRAD_H_
