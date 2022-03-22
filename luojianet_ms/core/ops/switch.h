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

#ifndef LUOJIANET_MS_CORE_OPS_SWITCH_H_
#define LUOJIANET_MS_CORE_OPS_SWITCH_H_
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace luojianet_ms {
namespace ops {
constexpr auto kNameSwitch = "Switch";
/// \brief Switch defined Switch operator prototype of lite.
class MS_CORE_API Switch : public PrimitiveC {
 public:
  /// \brief Constructor.
  Switch() : PrimitiveC(kNameSwitch) {}

  /// \brief Destructor.
  ~Switch() = default;

  MS_DECLARE_PARENT(Switch, PrimitiveC);
};
}  // namespace ops
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CORE_OPS_SWITCH_H_
