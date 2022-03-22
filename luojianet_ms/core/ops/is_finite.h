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

#ifndef LUOJIANET_MS_CORE_OPS_IS_FINITE_H_
#define LUOJIANET_MS_CORE_OPS_IS_FINITE_H_

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace luojianet_ms {
namespace ops {
constexpr auto kNameIsFinite = "IsFinite";
/// \brief Determines which elements are finite for each position.
/// Refer to Python API @ref luojianet_ms.ops.IsFinite for more details.
class MS_CORE_API IsFinite : public PrimitiveC {
 public:
  /// \brief Constructor.
  IsFinite() : PrimitiveC(kNameIsFinite) {}
  /// \brief Destructor.
  ~IsFinite() = default;
  MS_DECLARE_PARENT(IsFinite, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref luojianet_ms.ops.IsFinite for the inputs.
  void Init() {}
};
}  // namespace ops
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CORE_OPS_IS_FINITE_H_
