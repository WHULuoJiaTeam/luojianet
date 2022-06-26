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
#include "ops/arg_min.h"
#include "mindapi/ir/type.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
MIND_API_BASE_IMPL(ArgMin, PrimitiveC, BaseOperator);
void ArgMin::Init(const int64_t axis, const TypeId output_type) {
  set_axis(axis);
  set_output_type(output_type);
}

void ArgMin::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }
void ArgMin::set_output_type(const TypeId output_type) {
  (void)this->AddAttr(kOutputType, api::Type::GetType(output_type));
}

int64_t ArgMin::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }

TypeId ArgMin::get_output_type() const {
  auto type_ptr = GetAttr(kOutputType)->cast<api::TensorTypePtr>()->element();
  return type_ptr->type_id();
}

REGISTER_PRIMITIVE_C(kNameArgMin, ArgMin);
}  // namespace ops
}  // namespace luojianet_ms
