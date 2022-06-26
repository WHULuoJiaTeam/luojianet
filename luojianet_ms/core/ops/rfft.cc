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

#include "ops/rfft.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
void Rfft::Init(const int64_t fft_length) { this->set_fft_length(fft_length); }

void Rfft::set_fft_length(const int64_t fft_length) { (void)this->AddAttr(kFftLength, api::MakeValue(fft_length)); }

int64_t Rfft::get_fft_length() const { return GetValue<int64_t>(GetAttr(kFftLength)); }

MIND_API_BASE_IMPL(Rfft, PrimitiveC, BaseOperator);
REGISTER_PRIMITIVE_C(kNameRfft, Rfft);
}  // namespace ops
}  // namespace luojianet_ms
