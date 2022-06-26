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
#include "ops/cumsum.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
MIND_API_BASE_IMPL(CumSum, PrimitiveC, BaseOperator);
void CumSum::Init(const bool exclusive, const bool reverse) {
  this->set_exclusive(exclusive);
  this->set_reverse(reverse);
}

void CumSum::set_exclusive(const bool exclusive) { (void)this->AddAttr(kExclusive, api::MakeValue(exclusive)); }

bool CumSum::get_exclusive() const {
  auto value_ptr = this->GetAttr(kExclusive);
  return GetValue<bool>(value_ptr);
}

void CumSum::set_reverse(const bool reverse) { (void)this->AddAttr(kReverse, api::MakeValue(reverse)); }

bool CumSum::get_reverse() const {
  auto value_ptr = this->GetAttr(kReverse);
  return GetValue<bool>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameCumSum, CumSum);
}  // namespace ops
}  // namespace luojianet_ms
