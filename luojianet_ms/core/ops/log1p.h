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

#ifndef LUOJIANET_MS_CORE_OPS_LOG1P_H_
#define LUOJIANET_MS_CORE_OPS_LOG1P_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace luojianet_ms {
namespace ops {
constexpr auto kNameLog1p = "Log1p";

/// \brief Returns the natural logarithm of one plus the input tensor element-wise.
/// Refer to Python API @ref luojianet_ms.ops.Log1p for more details.
class MIND_API Log1p : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Log1p);
  /// \brief Constructor.
  Log1p() : BaseOperator(kNameLog1p) { InitIOName({"x"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref luojianet_ms.ops.Log1p for the inputs.
  void Init() const {}
};
}  // namespace ops
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CORE_OPS_LOG1P_H_
