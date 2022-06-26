/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ops/gru.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_BASE_IMPL(GRU, PrimitiveC, BaseOperator);
void GRU::Init(bool bidirectional) { this->set_bidirectional(bidirectional); }

void GRU::set_bidirectional(bool bidirectional) { (void)AddAttr(kBidirectional, api::MakeValue(bidirectional)); }

bool GRU::get_bidirectional() const {
  auto value_ptr = this->GetAttr(kBidirectional);
  return GetValue<bool>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameGRU, GRU);
}  // namespace ops
}  // namespace mindspore
