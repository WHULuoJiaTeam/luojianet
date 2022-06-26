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

#include "ops/uniform_real.h"
#include <string>
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
MIND_API_BASE_IMPL(UniformReal, PrimitiveC, BaseOperator);
void UniformReal::Init(int64_t seed, int64_t seed2) {
  this->set_seed(seed);
  this->set_seed2(seed2);
}

void UniformReal::set_seed(int64_t seed) { (void)this->AddAttr(kSeed, api::MakeValue(seed)); }

void UniformReal::set_seed2(int64_t seed2) { (void)this->AddAttr(kSeed2, api::MakeValue(seed2)); }

int64_t UniformReal::get_seed() const {
  auto value_ptr = GetAttr(kSeed);
  return GetValue<int64_t>(value_ptr);
}

int64_t UniformReal::get_seed2() const {
  auto value_ptr = GetAttr(kSeed2);
  return GetValue<int64_t>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameUniformReal, UniformReal);
}  // namespace ops
}  // namespace luojianet_ms
