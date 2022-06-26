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

#include "ops/control_depend.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
MIND_API_BASE_IMPL(ControlDepend, PrimitiveC, BaseOperator);
void ControlDepend::Init(const int64_t depend_mode) { this->set_depend_mode(depend_mode); }

void ControlDepend::set_depend_mode(const int64_t depend_mode) {
  CheckAndConvertUtils::CheckInRange<int64_t>(kDependMode, depend_mode, kIncludeBoth, {0, 1}, name());
  (void)AddAttr(kDependMode, api::MakeValue(depend_mode));
}

int64_t ControlDepend::get_depend_mode() const {
  auto value_ptr = GetAttr(kDependMode);
  return GetValue<int64_t>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameControlDepend, ControlDepend);
}  // namespace ops
}  // namespace luojianet_ms
