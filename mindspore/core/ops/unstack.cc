/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ops/unstack.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_BASE_IMPL(Unstack, PrimitiveC, BaseOperator);
void Unstack::Init(const int64_t axis) { this->set_axis(axis); }
void Unstack::set_axis(const int64_t axis) { (void)AddAttr(kAxis, api::MakeValue(axis)); }
int64_t Unstack::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }

REGISTER_PRIMITIVE_C(kNameUnstack, Unstack);
}  // namespace ops
}  // namespace mindspore
