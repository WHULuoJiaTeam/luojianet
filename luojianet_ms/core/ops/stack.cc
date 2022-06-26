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

#include "ops/stack.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
void Stack::set_axis(const int64_t axis) { (void)AddAttr(kAxis, api::MakeValue(axis)); }

int64_t Stack::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }

void Stack::Init(const int64_t axis) { this->set_axis(axis); }

MIND_API_BASE_IMPL(Stack, PrimitiveC, BaseOperator);
REGISTER_PRIMITIVE_C(kNameStack, Stack);
}  // namespace ops
}  // namespace luojianet_ms
