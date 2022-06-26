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
#ifndef LUOJIANET_MS_CORE_OPS_RELU_H_
#define LUOJIANET_MS_CORE_OPS_RELU_H_
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace luojianet_ms {
namespace ops {
constexpr auto kNameReLU = "ReLU";
/// \brief Computes ReLU (Rectified Linear Unit activation function) of input tensors element-wise.
/// Refer to Python API @ref luojianet_ms.ops.ReLU for more details.
class MIND_API ReLU : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ReLU);
  /// \brief Constructor.
  ReLU() : BaseOperator(kNameReLU) { InitIOName({"x"}, {"output"}); }
  /// \brief Init.
  void Init() const {}
};
}  // namespace ops
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CORE_OPS_RELU_H_
