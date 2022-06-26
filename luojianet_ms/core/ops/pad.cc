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
#include "ops/pad.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
void Pad::Init(const std::vector<std::vector<int64_t>> &paddings) { this->set_paddings(paddings); }
void Pad::set_paddings(const std::vector<std::vector<int64_t>> &paddings) {
  (void)this->AddAttr(kPaddings, api::MakeValue(paddings));
}
std::vector<std::vector<int64_t>> Pad::get_paddings() const {
  return GetValue<std::vector<std::vector<int64_t>>>(GetAttr(kPaddings));
}

MIND_API_BASE_IMPL(Pad, PrimitiveC, BaseOperator);
REGISTER_PRIMITIVE_C(kNamePad, Pad);
}  // namespace ops
}  // namespace luojianet_ms
