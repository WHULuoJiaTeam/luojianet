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

#include <string>
#include <map>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/expand_dims.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
MIND_API_BASE_IMPL(ExpandDims, PrimitiveC, BaseOperator);
REGISTER_PRIMITIVE_C(kNameExpandDims, ExpandDims);
}  // namespace ops
}  // namespace luojianet_ms
