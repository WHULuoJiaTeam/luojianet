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
#include <map>
#include <string>

#include "ops/grad/binary_cross_entropy_grad.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
void BinaryCrossEntropyGrad::Init(const Reduction &reduction) { set_reduction(reduction); }

void BinaryCrossEntropyGrad::set_reduction(const Reduction &reduction) {
  int64_t swi = reduction;
  (void)this->AddAttr(kReduction, api::MakeValue(swi));
}
Reduction BinaryCrossEntropyGrad::get_reduction() const {
  auto value_ptr = GetAttr(kReduction);
  return Reduction(GetValue<int64_t>(value_ptr));
}

MIND_API_BASE_IMPL(BinaryCrossEntropyGrad, PrimitiveC, BaseOperator);
REGISTER_PRIMITIVE_C(kNameBinaryCrossEntropyGrad, BinaryCrossEntropyGrad);
}  // namespace ops
}  // namespace luojianet_ms
